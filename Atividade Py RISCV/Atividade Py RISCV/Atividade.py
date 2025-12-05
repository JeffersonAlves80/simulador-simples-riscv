#!/usr/bin/env python3
"""
Simulador RISC-V RV32I - Implementação em Python
Arquivo único: riscv_sim.py

Funcionalidades:
- Decodificação e execução de RV32I (R, I, S, B, U, J)
- Registradores x0..x31 (x0 sempre zero)
- Barramento de 32 bits simples
- Memória mapeada:
    0x00000 - 0x7FFFF : RAM principal
    0x80000 - 0x8FFFF : VRAM (output via terminal)
    0x90000 - 0x9FBFF : Reservado
    0x9FC00 - 0x9FFFF : Periféricos (E/S mapeada)
- VRAM é escrita via SB/SW; a função `dump_vram()` mostra ASCII da área.
- Interrupção externa simulada por set_interrupt()
- Instrução JALR/JAL/JALL suportadas, branches, loads/stores, ALU ops.
- Programa de teste que escreve "HELLO\n" na VRAM e termina com ecall (ecall aqui termina).
"""

from typing import List
import struct
import sys

# -------------------------
# Helpers
# -------------------------
MASK32 = 0xFFFFFFFF

def sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (value & ((1 << bits) - 1)) - (sign_bit if (value & sign_bit) else 0)

def u32(x: int) -> int:
    return x & MASK32

def to_uint32(x: int) -> int:
    return x & 0xFFFFFFFF

# -------------------------
# Memory / Bus
# -------------------------

class Memory:
    def __init__(self, size_bytes=0xA0000):  # covers up to 0x9FFFF by default
        self.size = size_bytes
        self.mem = bytearray(self.size)
        # optionally pre-load with zeros (already zeroed)
    def check_addr(self, addr):
        if addr < 0 or addr >= self.size:
            raise Exception(f"Memory access out of range: 0x{addr:08x}")
    def load_u8(self, addr):
        self.check_addr(addr)
        return self.mem[addr]
    def load_i8(self, addr):
        return sign_extend(self.load_u8(addr), 8)
    def load_u16(self, addr):
        self.check_addr(addr+1)
        return self.mem[addr] | (self.mem[addr+1] << 8)
    def load_i16(self, addr):
        return sign_extend(self.load_u16(addr), 16)
    def load_u32(self, addr):
        self.check_addr(addr+3)
        return (self.mem[addr] |
                (self.mem[addr+1] << 8) |
                (self.mem[addr+2] << 16) |
                (self.mem[addr+3] << 24))
    def store_u8(self, addr, val):
        self.check_addr(addr)
        self.mem[addr] = val & 0xFF
    def store_u16(self, addr, val):
        self.check_addr(addr+1)
        self.mem[addr] = val & 0xFF
        self.mem[addr+1] = (val >> 8) & 0xFF
    def store_u32(self, addr, val):
        self.check_addr(addr+3)
        self.mem[addr] = val & 0xFF
        self.mem[addr+1] = (val >> 8) & 0xFF
        self.mem[addr+2] = (val >> 16) & 0xFF
        self.mem[addr+3] = (val >> 24) & 0xFF

# -------------------------
# Bus & Devices (simple)
# -------------------------
class Bus:
    def __init__(self, mem: Memory):
        self.mem = mem
        self.VRAM_START = 0x80000
        self.VRAM_SIZE = 0x10000  # 0x80000 - 0x8FFFF
        # peripheral range start
        self.PERIF_START = 0x9FC00
        self.interrupt_flag = False
    def load_u8(self, addr):
        return self.mem.load_u8(addr)
    def load_i8(self, addr):
        return self.mem.load_i8(addr)
    def load_u16(self, addr):
        return self.mem.load_u16(addr)
    def load_u32(self, addr):
        return self.mem.load_u32(addr)
    def store_u8(self, addr, val):
        self.mem.store_u8(addr, val)
    def store_u16(self, addr, val):
        self.mem.store_u16(addr, val)
    def store_u32(self, addr, val):
        self.mem.store_u32(addr, val)
    def is_vram_addr(self, addr):
        return self.VRAM_START <= addr < (self.VRAM_START + self.VRAM_SIZE)
    def set_interrupt(self):
        self.interrupt_flag = True
    def clear_interrupt(self):
        self.interrupt_flag = False
    def check_interrupt(self):
        return self.interrupt_flag

# -------------------------
# CPU (RV32I core)
# -------------------------
class CPU:
    def __init__(self, bus: Bus, pc_start=0x0, vram_refresh_interval=1000):
        self.bus = bus
        self.pc = pc_start
        self.regs = [0]*32
        self.cycle = 0
        self.vram_interval = vram_refresh_interval
        self.halted = False
        # CSRs minimal (not full)
        self.mepc = 0
        self.mcause = 0
    def fetch(self):
        instr = self.bus.load_u32(self.pc)
        return instr
    def read_reg(self, idx):
        if idx == 0:
            return 0
        return self.regs[idx] & MASK32
    def write_reg(self, idx, val):
        if idx == 0:
            return
        self.regs[idx] = val & MASK32
    def step(self):
        if self.halted:
            return
        instr = self.fetch()
        self.execute(instr)
        self.cycle += 1
        # VRAM refresh policy
        if self.vram_interval > 0 and (self.cycle % self.vram_interval == 0):
            dump_vram(self.bus)
        # simple interrupt handling
        if self.bus.check_interrupt():
            self.handle_interrupt()
    def run(self, max_steps=10_000_000):
        steps = 0
        try:
            while not self.halted and steps < max_steps:
                self.step()
                steps += 1
        except Exception as e:
            print(f"[SIM ERROR] {e}")
            self.halted = True
        return steps
    # --------------------------------
    # Interrupt (very simple)
    def handle_interrupt(self):
        # Save state and jump to address 0x1000 (for example) if exists
        self.mepc = self.pc
        self.mcause = 1
        # example interrupt vector
        IVT = 0x1000
        self.pc = IVT
        self.bus.clear_interrupt()
    # --------------------------------
    # Instruction decode / execute (RV32I)
    def execute(self, instr):
        pc_old = self.pc
        opcode = instr & 0x7f
        rd = (instr >> 7) & 0x1f
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1f
        rs2 = (instr >> 20) & 0x1f
        funct7 = (instr >> 25) & 0x7f

        def imm_i():
            return sign_extend((instr >> 20) & 0xFFF, 12)
        def imm_s():
            imm = ((instr >> 7) & 0x1f) | (((instr >> 25) & 0x7f) << 5)
            return sign_extend(imm, 12)
        def imm_b():
            imm = (((instr >> 8) & 0x0f) << 1) | (((instr >> 25) & 0x3f) << 5) | (((instr >> 7) & 0x1) << 11) | (((instr >> 31) & 0x1) << 12)
            return sign_extend(imm, 13)
        def imm_u():
            return (instr & 0xFFFFF000)
        def imm_j():
            imm = (((instr >> 21) & 0x3FF) << 1) | (((instr >> 20) & 0x1) << 11) | (((instr >> 12) & 0xFF) << 12) | (((instr >> 31) & 0x1) << 20)
            return sign_extend(imm, 21)

        next_pc = (self.pc + 4) & MASK32

        # R-type
        if opcode == 0x33:
            # ALU register
            if funct3 == 0x0 and funct7 == 0x00:   # add
                res = (self.read_reg(rs1) + self.read_reg(rs2)) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x0 and funct7 == 0x20: # sub
                res = (self.read_reg(rs1) - self.read_reg(rs2)) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x1 and funct7 == 0x00: # sll
                sh = self.read_reg(rs2) & 0x1F
                res = (self.read_reg(rs1) << sh) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x2 and funct7 == 0x00: # slt
                res = 1 if sign_extend(self.read_reg(rs1),32) < sign_extend(self.read_reg(rs2),32) else 0
                self.write_reg(rd, res)
            elif funct3 == 0x3 and funct7 == 0x00: # sltu
                res = 1 if self.read_reg(rs1) < self.read_reg(rs2) else 0
                self.write_reg(rd, res)
            elif funct3 == 0x4 and funct7 == 0x00: # xor
                res = self.read_reg(rs1) ^ self.read_reg(rs2)
                self.write_reg(rd, res)
            elif funct3 == 0x5 and funct7 == 0x00: # srl
                sh = self.read_reg(rs2) & 0x1F
                res = (self.read_reg(rs1) >> sh) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x5 and funct7 == 0x20: # sra
                sh = self.read_reg(rs2) & 0x1F
                res = (sign_extend(self.read_reg(rs1),32) >> sh) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x6 and funct7 == 0x00: # or
                res = self.read_reg(rs1) | self.read_reg(rs2)
                self.write_reg(rd, res)
            elif funct3 == 0x7 and funct7 == 0x00: # and
                res = self.read_reg(rs1) & self.read_reg(rs2)
                self.write_reg(rd, res)
            else:
                raise Exception(f"Unsupported R-type: funct7=0x{funct7:x} funct3=0x{funct3:x}")
        # I-type (loads, ALU imm, jalr)
        elif opcode == 0x13:
            # immediate ALU
            imm = imm_i()
            if funct3 == 0x0:  # addi
                res = (self.read_reg(rs1) + imm) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x2:  # slti
                res = 1 if sign_extend(self.read_reg(rs1),32) < imm else 0
                self.write_reg(rd, res)
            elif funct3 == 0x3:  # sltiu
                res = 1 if self.read_reg(rs1) < (imm & MASK32) else 0
                self.write_reg(rd, res)
            elif funct3 == 0x4:  # xori
                res = self.read_reg(rs1) ^ (imm & MASK32)
                self.write_reg(rd, res)
            elif funct3 == 0x6:  # ori
                res = self.read_reg(rs1) | (imm & MASK32)
                self.write_reg(rd, res)
            elif funct3 == 0x7:  # andi
                res = self.read_reg(rs1) & (imm & MASK32)
                self.write_reg(rd, res)
            elif funct3 == 0x1:  # slli (funct7 = 0)
                sh = imm & 0x1F
                res = (self.read_reg(rs1) << sh) & MASK32
                self.write_reg(rd, res)
            elif funct3 == 0x5:
                sh = imm & 0x1F
                if (imm >> 10) == 0:  # SRLI
                    res = (self.read_reg(rs1) >> sh) & MASK32
                else:  # SRAI
                    res = (sign_extend(self.read_reg(rs1),32) >> sh) & MASK32
                self.write_reg(rd, res)
            else:
                raise Exception(f"Unsupported I-type ALU funct3=0x{funct3:x}")
        elif opcode == 0x03:
            # Loads
            imm = imm_i()
            addr = (self.read_reg(rs1) + imm) & MASK32
            if funct3 == 0x0:  # LB
                val = self.bus.load_i8(addr)
                self.write_reg(rd, val & MASK32)
            elif funct3 == 0x1:  # LH
                val = self.bus.load_i16(addr)
                self.write_reg(rd, val & MASK32)
            elif funct3 == 0x2:  # LW
                val = self.bus.load_u32(addr)
                self.write_reg(rd, val & MASK32)
            elif funct3 == 0x4:  # LBU
                val = self.bus.load_u8(addr)
                self.write_reg(rd, val & MASK32)
            elif funct3 == 0x5:  # LHU
                val = self.bus.load_u16(addr)
                self.write_reg(rd, val & MASK32)
            else:
                raise Exception(f"Unsupported load funct3=0x{funct3:x}")
        elif opcode == 0x67:
            # JALR
            imm = imm_i()
            t = self.read_reg(rs1)
            target = (t + imm) & ~1
            self.write_reg(rd, next_pc)
            next_pc = target & MASK32
        # S-type (stores)
        elif opcode == 0x23:
            imm = imm_s()
            addr = (self.read_reg(rs1) + imm) & MASK32
            if funct3 == 0x0:  # SB
                self.bus.store_u8(addr, self.read_reg(rs2) & 0xFF)
            elif funct3 == 0x1:  # SH
                self.bus.store_u16(addr, self.read_reg(rs2) & 0xFFFF)
            elif funct3 == 0x2:  # SW
                self.bus.store_u32(addr, self.read_reg(rs2) & MASK32)
            else:
                raise Exception(f"Unsupported store funct3=0x{funct3:x}")
        # B-type (branches)
        elif opcode == 0x63:
            imm = imm_b()
            a = self.read_reg(rs1)
            b = self.read_reg(rs2)
            take = False
            if funct3 == 0x0:  # BEQ
                take = (a == b)
            elif funct3 == 0x1:  # BNE
                take = (a != b)
            elif funct3 == 0x4:  # BLT
                take = (sign_extend(a,32) < sign_extend(b,32))
            elif funct3 == 0x5:  # BGE
                take = (sign_extend(a,32) >= sign_extend(b,32))
            elif funct3 == 0x6:  # BLTU
                take = (a < b)
            elif funct3 == 0x7:  # BGEU
                take = (a >= b)
            if take:
                next_pc = (self.pc + imm) & MASK32
        # U-type (LUI/AUIPC)
        elif opcode == 0x37:  # LUI
            self.write_reg(rd, imm_u())
        elif opcode == 0x17:  # AUIPC
            self.write_reg(rd, (self.pc + imm_u()) & MASK32)
        # J-type (JAL)
        elif opcode == 0x6F:
            imm = imm_j()
            self.write_reg(rd, next_pc)
            next_pc = (self.pc + imm) & MASK32
        # SYSTEM (ecall)
        elif opcode == 0x73:
            if instr == 0x00000073:
                # ecall
                # We'll treat ecall as program termination for this simulator
                self.halted = True
            else:
                raise Exception(f"Unsupported SYSTEM instr 0x{instr:08x}")
        else:
            raise Exception(f"Unsupported opcode 0x{opcode:02x} at PC=0x{self.pc:08x}")

        # commit PC
        self.pc = next_pc

# -------------------------
# Utility: Dump VRAM
# -------------------------
def dump_vram(bus: Bus, start=0x80000, length=256):
    # print ASCII from VRAM region until null or length
    out = []
    for i in range(length):
        addr = start + i
        try:
            b = bus.load_u8(addr)
        except Exception:
            break
        if b == 0:
            break
        out.append(chr(b))
    s = ''.join(out)
    print(f"[VRAM] {s}")
    return s

# -------------------------
# Assembler-lite: create binary words from an array of 32-bit ints
# -------------------------
def load_program_words(mem: Memory, base: int, words: List[int]):
    addr = base
    for w in words:
        mem.store_u32(addr, w)
        addr += 4

# -------------------------
# Small helpers to encode instructions for test programs
# Only used to prepare the test program below
# -------------------------
def R(opcode, rd, funct3, rs1, rs2, funct7):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
def I(opcode, rd, funct3, rs1, imm):
    imm_u = imm & 0xFFF
    return (imm_u << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
def S(opcode, funct3, rs1, rs2, imm):
    imm_u = imm & 0xFFF
    imm_lo = imm_u & 0x1F
    imm_hi = (imm_u >> 5) & 0x7F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode
def B(opcode, funct3, rs1, rs2, imm):
    # imm is branch immediate (signed)
    imm_u = imm & 0x1FFF
    bit12 = (imm_u >> 12) & 0x1
    bit11 = (imm_u >> 11) & 0x1
    bits10_5 = (imm_u >> 5) & 0x3F
    bits4_1 = (imm_u >> 1) & 0xF
    return (bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (bits4_1 << 8) | (bit11 << 7) | opcode
def U(opcode, rd, imm):
    return (imm & 0xFFFFF000) | (rd << 7) | opcode
def J(opcode, rd, imm):
    imm_u = imm & 0x1FFFFF
    bit20 = (imm_u >> 20) & 0x1
    bits10_1 = (imm_u >> 1) & 0x3FF
    bits11 = (imm_u >> 11) & 0x1
    bits19_12 = (imm_u >> 12) & 0xFF
    return (bit20 << 31) | (bits19_12 << 12) | (bits11 << 20) | (bits10_1 << 21) | (rd << 7) | opcode

# -------------------------
# Program de teste: escreve "HELLO\n" na VRAM e encerra com ecall (0x00000073)
# Vamos colocar o programa em 0x0 na RAM.
# Estratégia:
# - Usar lui/addi para obter endereço VRAM base em x10
# - Escrever bytes com sb
# - Finalizar com ecall
# -------------------------
def build_test_program():
    words = []
    # constants
    VRAM_BASE = 0x80000
    # Use: auipc x10, offset; addi x10, x10, imm => x10 points to VRAM_BASE
    # For simplicity we'll use LUI to write upper bits, then addi
    hi = (VRAM_BASE + 0x800) & 0xFFFFF000  # choose conservative upper
    lo = VRAM_BASE - hi
    # LUI x10, hi
    words.append(U(0x37, 10, hi))              # LUI x10, hi
    # ADDI x10, x10, lo
    words.append(I(0x13, 10, 0x0, 10, lo))     # ADDI x10, x10, lo
    # Now store bytes for 'HELLO\n'
    s = b"HELLO\n"
    for i, ch in enumerate(s):
        # SB x11 = ch; to store we first load immediate into x11 via addi from x0
        words.append(I(0x13, 11, 0x0, 0, ch if ch < 2048 else (ch - 4096)))  # ADDI x11, x0, ch
        # SB x11, offset(x10)
        words.append(S(0x23, 0x0, 10, 11, i))  # SB
    # ecall
    words.append(0x00000073)
    return words

# -------------------------
# Entrypoint / Setup
# -------------------------
def main():
    print("RISC-V RV32I - Simulador (Python) - iniciando")
    mem = Memory()
    bus = Bus(mem)
    cpu = CPU(bus, pc_start=0x0, vram_refresh_interval=10)  # mostra VRAM a cada 10 instruções (ajustável)

    # load test program at address 0x0
    prog = build_test_program()
    load_program_words(mem, 0x0, prog)

    # run until ecall/halt
    steps = cpu.run(max_steps=100000)
    print(f"Executadas {steps} instruções")
    # final vram dump
    dump_vram(bus)

if __name__ == "__main__":
    main()
