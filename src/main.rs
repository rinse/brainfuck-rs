use anyhow::{anyhow, Context, Result};
use std::io::{Read, Stdin, Stdout, Write};

struct Machine<R: Read, W: Write> {
    program: Vec<u8>,
    program_counter: usize,
    memory: Vec<u8>,
    memory_ptr: usize,
    reader: R,
    writer: W,
}

fn machine_with_stdio(prog: Vec<u8>, mem: Vec<u8>) -> Machine<Stdin, Stdout> {
    Machine::new(prog, mem, std::io::stdin(), std::io::stdout())
}

impl<R: Read, W: Write> Machine<R, W> {
    pub fn new(prog: Vec<u8>, mem: Vec<u8>, r: R, w: W) -> Machine<R, W> {
        Machine {
            program: prog,
            program_counter: 0,
            memory: mem,
            memory_ptr: 0,
            reader: r,
            writer: w,
        }
    }

    fn _into_polymorphic<'a>(
        self: Machine<R, W>,
    ) -> Machine<Box<dyn 'a + Read>, Box<dyn 'a + Write>>
    where
        R: 'a,
        W: 'a,
    {
        Machine {
            program: self.program,
            program_counter: self.program_counter,
            memory: self.memory,
            memory_ptr: self.memory_ptr,
            reader: Box::new(self.reader),
            writer: Box::new(self.writer),
        }
    }

    pub fn run_program(&mut self) -> Result<()> {
        while self.program_counter < self.program.len() {
            self.step_run_program()?;
        }
        Ok(())
    }

    fn step_run_program(&mut self) -> Result<()> {
        let command = self.program[self.program_counter];
        self.process_command(command).context(format!(
            "Processing command error at :{}",
            self.program_counter
        ))
    }

    fn process_command(&mut self, command: u8) -> Result<()> {
        match command {
            0x2b => {
                // +
                self.increment_memory();
                self.program_counter += 1;
            }
            0x2d => {
                // -
                self.decrement_memory();
                self.program_counter += 1;
            }
            0x2c => {
                // ,
                self.read_char()?;
                self.program_counter += 1;
            }
            0x2e => {
                // .
                self.write_char()?;
                self.program_counter += 1;
            }
            0x3c => {
                // <
                self.backward_memory()?;
                self.program_counter += 1;
            }
            0x3e => {
                // >
                self.forward_memory()?;
                self.program_counter += 1;
            }
            0x5b => {
                // [
                self.forward_jump()?;
                self.program_counter += 1;
            }
            0x5d => {
                // ]
                self.backward_jump()?;
            }
            _ => {  // nop
                self.program_counter += 1
            }
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
        self.reader.read(&mut buf).context("Failed to read char!")?;
        self.memory[self.memory_ptr] = buf[0];
        Ok(())
    }

    fn write_char(&mut self) -> Result<()> {
        let current_value = self.memory[self.memory_ptr];
        self.writer
            .write(&[current_value; 1])
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
        if self.memory_ptr == 0 {
            return Err(anyhow!("Focus cannot be move backward anymore!"));
        }
        self.memory_ptr -= 1;
        Ok(())
    }

    fn forward_jump(&mut self) -> Result<()> {
        let current_value = self.memory[self.memory_ptr];
        if current_value != 0 {
            return Ok(());
        }
        let mut iter = self.program.iter().skip(self.program_counter);
        let mut brace_match = BraceMatch::new();
        find_first_index(&mut iter, |&&a| {
            brace_match.consume(a) && a == 0x5du8 // 0x5d is ']' on the ascii table
        })
        .map(|index| {
            self.program_counter += index;
        })
        .ok_or_else(|| anyhow!("Coresponding ']' not found!"))
    }

    fn backward_jump(&mut self) -> Result<()> {
        let mut iter = self.program.iter().take(self.program_counter + 1);
        let mut brace_match = BraceMatch::new();
        find_last_index(&mut iter, |&&a| {
            brace_match.consume(a) && a == 0x5bu8 // 0x5b is '[' on the ascii table
        })
        .map(|index| {
            self.program_counter = index;
        })
        .ok_or_else(|| anyhow!("Coresponding '[' not found!"))
    }
}

const CHUNK_SIZE: usize = 0x10;

fn display_bytes(bytes: &Vec<u8>, ptr: usize) -> String {
    let mut ret: String = String::with_capacity(bytes.len());
    let chunks = bytes.chunks(CHUNK_SIZE);
    for (i, chunk) in chunks.enumerate() {
        insert_chunk(i, chunk, &mut ret);
        ret.push('\n');
        insert_ptr(ptr, i, &mut ret);
    }
    ret
}

fn insert_chunk(line_number: usize, chunk: &[u8], out: &mut String) {
    out.push_str(&format!("{:04X}:", line_number));
    for n in chunk {
        out.push_str(&format!(" {:02X}", n));
    }
}

fn insert_ptr(ptr: usize, line_number: usize, out: &mut String) {
    let line_number_to_put = ptr / CHUNK_SIZE;
    if line_number_to_put != line_number {
        return;
    }
    let nth = ptr % CHUNK_SIZE; // memory_ptr pointes to the nth byte on the chunk
    out.push_str("     "); // header
    out.push_str(&"   ".repeat(nth)); // leading white spaces
    out.push_str(" ^^"); // indicator
    out.push_str("\n"); // newline
}

fn main() -> Result<()> {
    // let program = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).context("Failed to read from stdin")?;
    let mut machine = machine_with_stdio(input.as_bytes().to_vec(), vec![0; 0x10]);
    let r = machine.run_program();
    println!("hi");
    if let Err(e) = r {
        println!("{:?}", e);
    }
    println!("");
    println!("===memory dump===");
    print!("{}", display_bytes(&machine.memory, machine.memory_ptr));
    println!("memory_ptr: {}", machine.memory_ptr);
    println!("");
    println!("===program dump===");
    print!(
        "{}",
        display_bytes(&machine.program, machine.program_counter)
    );
    println!("program_counter: {}", machine.program_counter);
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
            0x5bu8 => self.count += 1, // [
            0x5du8 => self.count -= 1, // ]
            _ => {}
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
    P: FnMut(&T) -> bool,
{
    iter.enumerate().find(|(_, a)| predicate(a)).map(|(i, _)| i)
}

fn find_last_index<I, T, P>(iter: &mut I, mut predicate: P) -> Option<usize>
where
    I: Iterator<Item = T> + DoubleEndedIterator + ExactSizeIterator,
    P: FnMut(&T) -> bool,
{
    iter.enumerate()
        .rfind(|(_, a)| predicate(a))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{stdin, stdout, Write};

    #[test]
    fn increment_memory_increments_currently_focused_memory() {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        machine.increment_memory();
        assert_eq!(machine.memory, vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn increment_memory_wrapps_overflow() {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        for _ in 0..256 {
            machine.increment_memory();
        }
        assert_eq!(machine.memory, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn decrement_memory_increments_currently_focused_memory() {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        machine.increment_memory();
        machine.decrement_memory();
        assert_eq!(machine.memory, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn decrement_memory_wrapps_overflow() {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        machine.decrement_memory();
        assert_eq!(machine.memory, vec![255, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn read_char_reads_char() -> anyhow::Result<()> {
        let input = "hello".as_bytes();
        let mut machine = Machine::new(vec![], vec![0; 10], Box::new(input), Box::new(stdout()));
        machine.read_char()?;
        assert_eq!(machine.memory, vec![0x68, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn write_char_writes_char() -> anyhow::Result<()> {
        let mut machine = Machine::new(vec![], vec![0x65], Box::new(stdin()), Box::new(vec![]));
        machine.memory[0] = 0x65; // 'A'
        machine.write_char()?;
        machine.writer.flush()?;
        let output_buffer = *machine.writer;
        assert_eq!(output_buffer, vec![0x65u8]);
        Ok(())
    }

    #[test]
    fn forward_memory_moves_focus_forward() -> anyhow::Result<()> {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        machine.forward_memory()?;
        machine.increment_memory();
        assert_eq!(machine.memory, vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn forward_memory_returns_err_when_focus_cannot_be_move_forward() {
        let mut machine = machine_with_stdio(vec![], vec![0; 1]);
        let ret = machine.forward_memory();
        assert_eq!(ret.is_err(), true);
    }

    #[test]
    fn backward_memory_moves_focus_backward() -> anyhow::Result<()> {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        machine.forward_memory()?;
        machine.backward_memory()?;
        machine.increment_memory();
        assert_eq!(machine.memory, vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn backward_memory_returns_err_when_focus_cannot_be_move_backward() {
        let mut machine = machine_with_stdio(vec![], vec![0; 10]);
        let ret = machine.backward_memory();
        assert_eq!(ret.is_err(), true);
    }

    #[test]
    fn forward_jump_jumps_when_current_focused_value_is_0() -> anyhow::Result<()> {
        let program = "[++++++]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        machine.forward_jump()?;
        assert_eq!(machine.program_counter, 7);
        Ok(())
    }

    #[test]
    fn forward_jump_jumps_when_current_focused_value_is_0_nested() -> anyhow::Result<()> {
        let program = "[++[++++++]++]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        machine.forward_jump()?;
        assert_eq!(machine.program_counter, 13);
        Ok(())
    }

    #[test]
    fn forward_jump_jumps_returns_err_when_no_corresponding_brace_found() {
        let program = "[++++++";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        let r = machine.forward_jump();
        assert_eq!(r.is_err(), true);
    }

    #[test]
    fn backward_jump_jumps_to_corresponding_brace() -> anyhow::Result<()> {
        let program = "[++++++]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        machine.program_counter = 7;
        machine.backward_jump()?;
        assert_eq!(machine.program_counter, 0);
        Ok(())
    }

    #[test]
    fn backward_jump_jumps_to_corresponding_brace_nested() -> anyhow::Result<()> {
        let program = "[++[++++++]++]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        machine.program_counter = 13;
        machine.backward_jump()?;
        assert_eq!(machine.program_counter, 0);
        Ok(())
    }

    #[test]
    fn backward_jump_jumps_returns_err_when_no_corresponding_brace_found() {
        let program = "++++++]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        let r = machine.backward_jump();
        assert_eq!(r.is_err(), true);
    }

    #[test]
    fn test_brace_match_forward() {
        let mut b = BraceMatch::new();
        b.consume(0x5b);
        b.consume(0);
        b.consume(0x5d);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_forward_nested() {
        let mut b = BraceMatch::new();
        b.consume(0x5b);
        b.consume(0x5b);
        b.consume(0);
        b.consume(0x5d);
        b.consume(0x5d);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_backward() {
        let mut b = BraceMatch::new();
        b.consume(0x5d);
        b.consume(0);
        b.consume(0x5b);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_backward_nested() {
        let mut b = BraceMatch::new();
        b.consume(0x5d);
        b.consume(0x5d);
        b.consume(0);
        b.consume(0x5b);
        b.consume(0x5b);
        assert_eq!(b.is_balanced(), true);
    }

    #[test]
    fn test_brace_match_unmatched_forward() {
        let mut b = BraceMatch::new();
        b.consume(0x5b);
        assert_eq!(b.is_balanced(), false);
    }

    #[test]
    fn test_brace_match_unmatched_backward() {
        let mut b = BraceMatch::new();
        b.consume(0x5d);
        assert_eq!(b.is_balanced(), false);
    }

    #[test]
    fn find_first_index_finds_index_of_first_occurrence() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = find_first_index(&mut v.iter(), |&&a| a == 'b');
        assert_eq!(r, Some(1));
    }

    #[test]
    fn find_first_index_is_inclusive() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = find_first_index(&mut v.iter(), |&&a| a == 'a');
        assert_eq!(r, Some(0));
    }

    #[test]
    fn find_first_index_returns_none_for_not_found() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = find_first_index(&mut v.iter(), |&&a| a == 'z');
        assert_eq!(r, None);
    }

    #[test]
    fn find_last_index_finds_index_of_last_occurrence() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = find_last_index(&mut v.iter(), |&&a| a == 'b');
        assert_eq!(r, Some(3));
    }

    #[test]
    fn find_last_index_returns_none_for_not_found() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = find_last_index(&mut v.iter(), |&&a| a == 'z');
        assert_eq!(r, None);
    }

    #[test]
    fn find_last_index_is_inclusive() {
        let v = vec!['a', 'b', 'c', 'b', 'a'];
        let r = find_last_index(&mut v.iter(), |&&a| a == 'a');
        assert_eq!(r, Some(4));
    }

    #[test]
    fn run_program_increment() -> anyhow::Result<()> {
        let program = "+++++";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 3]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![5, 0, 0]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, program.len());
        Ok(())
    }

    #[test]
    fn run_program_decrement() -> anyhow::Result<()> {
        let program = "-----";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 3]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![251, 0, 0]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, program.len());
        Ok(())
    }

    #[test]
    fn run_program_move_forward() -> anyhow::Result<()> {
        let program = ">>>>>";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0; 10]);
        assert_eq!(machine.memory_ptr, 5);
        assert_eq!(machine.program_counter, program.len());
        Ok(())
    }

    #[test]
    fn run_program_move_backward() -> anyhow::Result<()> {
        let program = ">>>>><<<<<";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 10]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0; 10]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, program.len());
        Ok(())
    }

    #[test]
    fn run_program_loop_copy_memory() -> anyhow::Result<()> {
        let program = "++[->+<]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 5]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0, 2, 0, 0, 0]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, program.len());
        Ok(())
    }

    #[test]
    fn run_program_nested_loop() -> anyhow::Result<()> {
        // 0x01 and 0x02 are loop counters, 0x03 is a output.
        // 72 = 8 * 3 * 3
        let program = "++++++++[>+++[>+++<-]<-]";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 5]);
        machine.run_program()?;
        assert_eq!(machine.memory, vec![0, 0, 72, 0, 0]);
        assert_eq!(machine.memory_ptr, 0);
        assert_eq!(machine.program_counter, program.len());
        Ok(())
    }

    #[test]
    fn run_program_empty() -> anyhow::Result<()> {
        let program = "";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 0x10]);
        machine.run_program()?;
        Ok(())
    }

    #[test]
    fn run_program_ignores_unrecognized_commands() -> anyhow::Result<()> {
        let program = "abc";
        let mut machine = machine_with_stdio(program.as_bytes().to_vec(), vec![0; 0x10]);
        machine.run_program()?;
        Ok(())
    }

    #[test]
    fn run_program_hello_world() -> anyhow::Result<()> {
        let program = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
        let mut machine = Machine::new(program.as_bytes().to_vec(), vec![0; 0x10], stdin(), vec![]);
        machine.run_program()?;
        let memory: Vec<u8> = machine.writer;
        let s: &str = std::str::from_utf8(memory.as_slice()).context("Failed to encode as utf8")?;
        assert_eq!(s, "Hello World!\n");
        Ok(())
    }
}
