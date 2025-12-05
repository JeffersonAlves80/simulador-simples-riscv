# RISC-V RV32I Simulator (Python)

Este projeto implementa, em um único arquivo Python (`riscv_sim.py`), um **simulador completo do conjunto básico RV32I** da arquitetura **RISC-V**.
O foco é **simplicidade**, **legibilidade** e **funcionalidade educacional** — ideal para estudar como uma CPU RISC-V funciona internamente.

---

## ✨ Funcionalidades

### 🧠 CPU (RV32I)

* Suporte completo às instruções:

  * **R-type**
  * **I-type**
  * **S-type**
  * **B-type**
  * **U-type**
  * **J-type**
* Implementação de:

  * `ADD`, `SUB`, `AND`, `OR`, `XOR`, shifts, comparações
  * Loads e stores (`LB`, `LH`, `LW`, `SB`, `SH`, `SW`)
  * Desvios condicionais e incondicionais
  * `JAL`, `JALR`
  * `ECALL` (aqui usada para encerrar o programa)
* Registradores `x0` até `x31` (com `x0` sempre sendo 0)
* PC com incremento automático e lógica de salto
* Tratamento simples de interrupções (por `set_interrupt()`)

---

## 🗄️ Memória e Mapeamento

A memória é um array contínuo de 32 bits, com regiões mapeadas:

| Região                    | Endereço            | Descrição              |
| ------------------------- | ------------------- | ---------------------- |
| RAM principal             | `0x00000 - 0x7FFFF` | RAM normal             |
| VRAM                      | `0x80000 - 0x8FFFF` | Saída de vídeo (ASCII) |
| Área reservada            | `0x90000 - 0x9FBFF` | Reservado              |
| Periféricos (I/O mapeado) | `0x9FC00 - 0x9FFFF` | E/S simulada           |

### VRAM

A VRAM funciona como uma “tela” textual.
Quando o código executa um `SB` ou `SW` nessa região, o simulador pode exibir automaticamente o resultado via a função:

```
dump_vram()
```

Perfeito para testar prints em nível de assembly.

---

## 🔌 Barramento e Periféricos

* A classe `Bus` abstrai acesso de leitura/escrita à RAM, VRAM e dispositivos.
* Possui flag de interrupção simulada:

  * `set_interrupt()`
  * `clear_interrupt()`
  * CPU salta para o vetor fictício em `0x1000`

---

## 🧱 Estrutura do Arquivo

Tudo está contido em **um único arquivo**:

* Helpers (sign-extend e máscaras)
* Classe `Memory`
* Classe `Bus`
* Classe `CPU`
* Utilidades de assembler (para montar instruções de teste)
* Programa exemplo
* Função `main()` para rodar o simulador

---

## 🧪 Programa de Teste

O simulador já vem com um pequeno programa RV32I que:

1. Calcula o endereço base da VRAM
2. Escreve os caracteres `"HELLO\n"` byte a byte usando `SB`
3. Finaliza com `ECALL`

Quando executado, deve aparecer algo assim no console:

```
[VRAM] HELLO
Executadas XXXX instruções
[VRAM] HELLO
```

---

## ▶️ Como Executar

Requer Python 3.6+.

```bash
python3 riscv_sim.py
```

Nenhuma dependência externa é necessária.

---

## 📁 Estrutura do Projeto

```
riscv_sim.py   # Todo o simulador em um único arquivo
```

---

## 🎯 Objetivos do Projeto

Este simulador foi criado para:

* Entender o ciclo de execução de uma CPU RISC-V
* Ter uma implementação clara do conjunto RV32I
* Brincar com programas em assembly para rodar sobre a CPU simulada
* Facilitar experimentos com VRAM, branches e manipulação de memória

---

## 💡 Possíveis Expansões

* Suporte a RV32M (multiplicação e divisão)
* Implementação mais completa de CSRs
* Pipeline (5 estágios)
* Cache/L1 simulado
* Debugger passo a passo
* Carregamento de binários ELF

