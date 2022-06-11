use std::io::{Read, Write};
use anyhow::{Context, Result, anyhow};

struct Machine {
    program: Vec<u8>,
    program_counter: usize,
    memory: Vec<u8>,
    memory_ptr: usize,
    writer: Box<dyn Write>,
    reader: Box<dyn Read>,
}

impl Machine {
    pub fn new(prog: Vec<u8>, mem: Vec<u8>) -> Machine {
        Machine::new_rw(prog, mem, Box::new(std::io::stdin()), Box::new(std::io::stdout()))
    }

    pub fn new_rw(prog: Vec<u8>, mem: Vec<u8>, r: Box<dyn Read>, w: Box<dyn Write>) -> Machine {
        Machine {
            program: prog,
            program_counter: 0,
            memory: mem,
            memory_ptr: 0,
            writer: w,
            reader: r,
        }
    }

    pub fn run_program(&mut self) -> Result<()> {
        while self.program_counter < self.program.len() {
            self.run_program_step()?;
        }
        Ok(())
    }

    fn run_program_step(&mut self) -> Result<()> {
        let command = self.program[self.program_counter];
        self.process_command(command)
            .context(format!("Processing command error at :{}", self.program_counter))
    }

    fn process_command(&mut self, command: u8) -> Result<()> {
        match command {
            0x2b => {   // +
                self.increment_memory();
                self.program_counter += 1;
            },
            0x2d => {   // -
                self.decrement_memory();
                self.program_counter += 1;
            },
            0x2c => {   // ,
                self.read_char()?;
                self.program_counter += 1;
            },
            0x2e => {   // .
                self.write_char()?;
                self.program_counter += 1;
            },
            0x3c => {   // <
                self.backward_memory()?;
                self.program_counter += 1;
            },
            0x3e => {   // >
                self.forward_memory()?;
                self.program_counter += 1;
            },
            0x5b => {   // [
                self.forward_jump()?;
                self.program_counter += 1;
            },
            0x5d => {   // ]
                self.backward_jump()?;
            },
            _    => {}, // nop
        };
        Ok(())
    }

    fn increment_memory(&mut self) {
        let current_value = self.memory[self.memory_ptr];
        self.memory[self.memory_ptr] = current_value.wrapping_add(1);
    }

    fn decrement_memory(&mut self) {
        let current_value = self.memory[self.memory_ptr];
        self.memory[self.memory_ptr] = current_value.wrapping_sub(1);
    }

    fn read_char(&mut self) -> Result<()> {
        let mut buf: [u8; 1] = [0; 1];
        self.reader.read(&mut buf)
            .context("Failed to read char!")?;
        self.memory[self.memory_ptr] = buf[0];
        Ok(())
    }

    fn write_char(&mut self) -> Result<()> {
        let current_value = self.memory[self.memory_ptr];
        self.writer.write(&[current_value; 1])
            .map(|_| ())
            .context("Failed to write char!")
    }

    fn forward_memory(&mut self) -> Result<()> {
        if self.memory_ptr >= self.memory.len() - 1 {
            return Err(anyhow!("Focus cannot be move forward anymore!"));
        }
        self.memory_ptr += 1;
        Ok(())
    }

    fn backward_memory(&mut self) -> Result<()> {
        if self.memory_ptr <= 0 {
            return Err(anyhow!("Focus cannot be move backward anymore!"));
        }
        self.memory_ptr -= 1;
        Ok(())
    }

    fn forward_jump(&mut self) -> Result<()> {
        let current_value = self.memory[self.memory_ptr];
        if current_value != 0 {
            return Ok(())
        }
        let mut iter = self.program.iter().skip(self.program_counter);
        let mut brace_match = BraceMatch::new();
        find_first_index(&mut iter, |&&a| {
                brace_match.consume(a) && a == 0x5du8 // 0x5d is ']' on the ascii table
            })
            .map(|index| { self.program_counter = self.program_counter + index; })
            .ok_or(anyhow!("Coresponding ']' not found!"))
    }

    fn backward_jump(&mut self) -> Result<()> {
        let mut iter = self.program.iter().take(self.program_counter + 1);
        let mut brace_match = BraceMatch::new();
        find_last_index(&mut iter, |&&a| {
                brace_match.consume(a) && a == 0x5bu8 // 0x5b is '[' on the ascii table
            })
            .map(|index| { self.program_counter = index; })
            .ok_or(anyhow!("Coresponding '[' not found!"))
    }
}

fn display_memory(machine: &Machine) -> String {
    let mut r: String = String::new();
    r.push_str("0x: 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f\n");
    let chunks = machine.memory.chunks(0x10);
    for (i, chunk) in chunks.enumerate() {
        r.push_str(&display_memory_chunk(i, chunk));
    }
    return r;
}

fn display_memory_chunk(line_number: usize, chunk: &[u8]) -> String {
    let mut r: String = String::new();
    r.push_str(&format!("{:02X}:", line_number));
    for n in chunk {
        r.push_str(&format!(" {:02X}", n));
    }
    return r;
}

fn main() -> Result<()> {
    let program = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    let mut machine = Machine::new(program.as_bytes().to_vec(), vec![0; 0x10]);
    let r = machine.run_program();
    if let Err(e) = r {
        println!("{:?}", e);
        println!("===memory dump===");
        println!("{}", display_memory(&machine));
        println!("memory_ptr: {}", machine.memory_ptr);
        println!("program_counter: {}", machine.program_counter);
        unsafe {
            let c = machine.program[machine.program_counter];
            println!("program command: {}", std::char::from_u32_unchecked(c as u32));
        }
    }
    Ok(())
}

struct BraceMatch {
    count: i32,
}

impl BraceMatch {
    fn new() -> BraceMatch {
        BraceMatch { count: 0 }
    }

    fn consume(&mut self, c: u8) -> bool {
        match c {
            0x5bu8 => self.count += 1,  // [
            0x5du8 => self.count -= 1,  // ]
            _      => {},
        }
        self.is_balanced()
    }

    fn is_balanced(&self) -> bool {
        self.count == 0
    }
}

fn find_first_index<I, T, P>(iter: &mut I, mut predicate: P) -> Option<usize>
    where
        I: Iterator<Item = T>,
        P: FnMut (&T) -> bool, {
    iter.enumerate()
        .find(|(_, a)| predicate(a))
        .map(|(i, _)| i)
}

fn find_last_index<I, T, P>(iter: &mut I, mut predicate: P) -> Option<usize>
    where
        I: Iterator<Item = T> + DoubleEndedIterator + ExactSizeIterator,
        P: FnMut (&T) -> bool, {
    iter.enumerate()
        .rfind(|(_, a)| predicate(a))
        .map(|(i, _)| i)
}

mod tests {
    #[test]
    fn increment_memory_increments_currently_focused_memory() {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        machine.increment_memory();
        assert_eq!(machine.memory, vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn increment_memory_wrapps_overflow() {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        for _ in 0..256 {
            machine.increment_memory();
        }
        assert_eq!(machine.memory, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn decrement_memory_increments_currently_focused_memory() {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        machine.increment_memory();
        machine.decrement_memory();
        assert_eq!(machine.memory, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn decrement_memory_wrapps_overflow() {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        machine.decrement_memory();
        assert_eq!(machine.memory, vec![255, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn read_char_reads_char() -> anyhow::Result<()> {
        let input = "hello".as_bytes();
        let mut machine = super::Machine::new_rw(
            vec![], vec![0; 10],
            Box::new(input),
            Box::new(std::io::stdout()),
        );
        machine.read_char()?;
        assert_eq!(machine.memory, vec![0x68, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    // TODO
    // #[test]
    // fn write_char_writes_char() -> anyhow::Result<()> {
    //     let mut v: Vec<u8> = Vec::new();
    //     v.resize(10, 0);
    //     let mut machine = super::Machine::new_rw(
    //         vec![], vec![0x65],
    //         Box::new(std::io::stdin()),
    //         Box::new(v),
    //     );
    //     machine.memory[0] = 0x65;   // 'A'
    //     machine.write_char()?;
    //     assert_eq!(v[0], 0x65u8);
    //     Ok(())
    // }

    #[test]
    fn forward_memory_moves_focus_forward() -> anyhow::Result<()> {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        machine.forward_memory()?;
        machine.increment_memory();
        assert_eq!(machine.memory, vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn forward_memory_returns_err_when_focus_cannot_be_move_forward() {
        let mut machine = super::Machine::new(vec![], vec![0; 1]);
        let ret = machine.forward_memory();
        assert_eq!(ret.is_err(), true);
    }

    #[test]
    fn backward_memory_moves_focus_backward() -> anyhow::Result<()> {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        machine.forward_memory()?;
        machine.backward_memory()?;
        machine.increment_memory();
        assert_eq!(machine.memory, vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn backward_memory_returns_err_when_focus_cannot_be_move_backward() {
        let mut machine = super::Machine::new(vec![], vec![0; 10]);
        let ret = machine.backward_memory();
        assert_eq!(ret.is_err(), true);
    }

    #[test]
    fn forward_jump_jumps_when_current_focused_value_is_0() -> anyhow::Result<()> {
        let program = "[++++++]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        machine.forward_jump()?;
        assert_eq!(machine.program_counter, 7);
        Ok(())
    }

    #[test]
    fn forward_jump_jumps_when_current_focused_value_is_0_nested() -> anyhow::Result<()> {
        let program = "[++[++++++]++]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        machine.forward_jump()?;
        assert_eq!(machine.program_counter, 13);
        Ok(())
    }

    #[test]
    fn forward_jump_jumps_returns_err_when_no_corresponding_brace_found() {
        let program = "[++++++";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        let r = machine.forward_jump();
        assert_eq!(r.is_err(), true);
    }

    #[test]
    fn backward_jump_jumps_to_corresponding_brace() -> anyhow::Result<()> {
        let program = "[++++++]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        machine.program_counter = 7;
        machine.backward_jump()?;
        assert_eq!(machine.program_counter, 0);
        Ok(())
    }

    #[test]
    fn backward_jump_jumps_to_corresponding_brace_nested() -> anyhow::Result<()> {
        let program = "[++[++++++]++]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        machine.program_counter = 13;
        machine.backward_jump()?;
        assert_eq!(machine.program_counter, 0);
        Ok(())
    }

    #[test]
    fn backward_jump_jumps_returns_err_when_no_corresponding_brace_found() {
        let program = "++++++]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        let r = machine.backward_jump();
        assert_eq!(r.is_err(), true);
    }

    #[test]
    fn test_brace_match_forward() {
        let mut b = super::BraceMatch::new();
        b.consume(0x5b);
        b.consume(0);
        b.consume(0x5d);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_forward_nested() {
        let mut b = super::BraceMatch::new();
        b.consume(0x5b);
        b.consume(0x5b);
        b.consume(0);
        b.consume(0x5d);
        b.consume(0x5d);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_backward() {
        let mut b = super::BraceMatch::new();
        b.consume(0x5d);
        b.consume(0);
        b.consume(0x5b);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_backward_nested() {
        let mut b = super::BraceMatch::new();
        b.consume(0x5d);
        b.consume(0x5d);
        b.consume(0);
        b.consume(0x5b);
        b.consume(0x5b);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_unmatched_forward() {
        let mut b = super::BraceMatch::new();
        b.consume(0x5b);
        assert_eq!(b.is_balanced(), false);
    }

    #[test]
    fn test_brace_match_unmatched_backward() {
        let mut b = super::BraceMatch::new();
        b.consume(0x5d);
        assert_eq!(b.is_balanced(), false);
    }

    #[test]
    fn find_first_index_finds_index_of_first_occurrence() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = super::find_first_index(&mut v.iter(), |&&a| a == 'b');
        assert_eq!(r, Some(1));
    }

    #[test]
    fn find_first_index_is_inclusive() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = super::find_first_index(&mut v.iter(), |&&a| a == 'a');
        assert_eq!(r, Some(0));
    }

    #[test]
    fn find_first_index_returns_none_for_not_found() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = super::find_first_index(&mut v.iter(), |&&a| a == 'z');
        assert_eq!(r, None);
    }

    #[test]
    fn find_last_index_finds_index_of_last_occurrence() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = super::find_last_index(&mut v.iter(), |&&a| a == 'b');
        assert_eq!(r, Some(3));
    }

    #[test]
    fn find_last_index_returns_none_for_not_found() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = super::find_last_index(&mut v.iter(), |&&a| a == 'z');
        assert_eq!(r, None);
    }

    #[test]
    fn find_last_index_is_inclusive() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = super::find_last_index(&mut v.iter(), |&&a| a == 'a');
        assert_eq!(r, Some(4));
    }

    #[test]
    fn run_program_simple1() -> anyhow::Result<()> {
        let program = "+++++-----";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0; 10]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, 10);
        Ok(())
    }

    #[test]
    fn run_program_simple2() -> anyhow::Result<()> {
        let program = ">>>>><<<<<";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 10]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0; 10]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, 10);
        Ok(())
    }


    #[test]
    fn run_program_simple_loop1() -> anyhow::Result<()> {
        let program = "[->+<]++";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 5]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![2, 0, 0, 0, 0]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, 8);
        Ok(())
    }

    #[test]
    fn run_program_simple_loop2() -> anyhow::Result<()> {
        let program = "++[->+<]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 5]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0, 2, 0, 0, 0]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, 8);
        Ok(())
    }

    #[test]
    fn run_program_hello_world() -> anyhow::Result<()> {
        let program = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]";
        let mut machine = super::Machine::new(program.as_bytes().to_vec(), vec![0; 7]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0, 0, 72, 104, 88, 32, 8]);
        Ok(())
    }
}
