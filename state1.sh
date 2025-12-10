#!/bin/bash

PROMPT=$(
  cat <<'EOF'
Genera el código Verilog completo y sintetizable para una Máquina de Estados Finitos (FSM) tipo Mealy que controla una lavadora con sistema de crédito.

Estados (2 bits): S0=00 (reposo, $0), S1=01 ($1), S2=10 ($2), S3=11 ($3 máximo)

Entradas: clk, rst_n (reset asíncrono activo bajo), K (Start), C[1:0] (crédito)

Salidas: A (aceptado), P[1:0] (programa: 01=Normal, 10=Delicado, 11=Especial)

Transiciones:
- S0 + C=01 sin K -> S1
- S1 + C=01 sin K -> S2  
- S2 + C=01 sin K -> S3
- Cualquier estado + K con crédito válido -> S0, A=1, P=crédito actual

Ecuaciones SOP:
S'0 = (~S1 & ~S0 & ~K & ~C1 & C0) | (~S1 & S0 & ~K & ~C1 & C0) | (S1 & ~S0 & ~K & C1 & C0) | (S1 & S0 & ~K & C1 & C0)
S'1 = (~S1 & S0 & ~K & C1 & ~C0) | (S1 & ~S0 & ~K & C1 & ~C0) | (S1 & ~S0 & ~K & C1 & C0) | (S1 & S0 & ~K & C1 & C0)
A = (~S1 & ~S0 & K & ~C1 & C0) | (~S1 & S0 & K & ~C1 & C0) | (~S1 & S0 & K & C1 & ~C0) | (S1 & ~S0 & K & C1 & ~C0) | (S1 & ~S0 & K & C1 & C0) | (S1 & S0 & K & C1 & C0)
P0 = (~S1 & ~S0 & K & ~C1 & C0) | (~S1 & S0 & K & ~C1 & C0) | (S1 & ~S0 & K & C1 & C0) | (S1 & S0 & K & C1 & C0)
P1 = (~S1 & S0 & K & C1 & ~C0) | (S1 & ~S0 & K & C1 & ~C0) | (S1 & ~S0 & K & C1 & C0) | (S1 & S0 & K & C1 & C0)

Usa estilo de 3 bloques always, localparam para estados, case statements con default. Incluye testbench.
EOF
)

mlx_lm.generate --model Qwen3-4B-Thinking-2507-MLX-4bit --adapter-path adapters \
  --prompt "$PROMPT" --max-tokens 2000000
