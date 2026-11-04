;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_hybrid_104b.asm)
; Description: 104-byte Multi-Stage Hybrid 2D x86-16 decompressor (-4B vs 108B, <= 128B budget)
;              unifying 0B-Overhead ZX0 v2, 1-pass or 2-pass x86 E8/E9 rel16 CALL/JMP restoration,
;              and Free-BX Multi-Stride filtering (1..256) via algebraic EOF dispatch:
;              - Mode 0 (V=256): Standard ZX0 v2 EOF (CX=0xFF00; at .eof: 'inc ch' sets ZF=1 -> ret),
;                achieving 100% bit-for-bit backward compatibility with standard ZX0 v2 streams and
;                ZERO bytes of header/trailer overhead on unfiltered data.
;              - Mode 1 (V=257): Free-BX Multi-Pass 2D Stride-Delta (CH=0xFE -> inc ch -> CH=0xFF,
;                JS=0: enters .mp_loop directly; 'inc ax / js .e8_loop' is not taken).
;                Reads an explicit 1-byte negative stride ('lodsb / mov bl, al / mov bh, 0xff'),
;                freeing the DP parser to choose optimal match offsets up to EOF, while each
;                integration pass can use its own independent stride S_p in 1..256.
;              - Mode 2 (V=1049 / V=1048): 1-pass x86 E8 CALL rel16 (V=1049) or E9 JMP rel16 (V=1048)
;                filter (CH=0xFB -> inc ch -> CH=0xFC; after .e8_loop, 'ror ah, 1' -> 0xFE -> inc ch
;                -> 0xFF), followed automatically by K optional passes of 2D Stride-Delta.
;              - Mode 3 (V=537): 2-pass Chained x86 E8 + E9 CALL/JMP rel16 filter (CH=0xFD -> inc ch
;                -> CH=0xFE; Pass 1 decodes AL=0xE8, 'ror ah, 1 / inc ch' yields AX=0xFFE8 so Pass 2
;                'inc ax' decodes AL=0xE9, and Pass 3+ transitions automatically into .floop).
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Multi-stage hybrid E8/E9 + Free-BX Stride size-coding (104B): Oleksandr Dreval (MIT License)
;
; License: MIT (SPDX-License-Identifier: MIT)
;--------------------------------------------------------------------------------------------------

bits 16

dzx0:
    mov al, 0x80
    dec bx

.literals:
    call .get_elias
    rep movsb
    add al, al
    jc .get_offset
    call .get_elias

.do_copy:
    push si
    lea si, [di+bx]
    rep movsb
    pop si
    add al, al
    jnc .literals

.get_offset:
    mov cl, 0xfe
    call .elias_loop
    inc cx
    js .eof

    mov bh, cl
    mov cx, 1
    mov bl, [si]
    inc si
    rcr bx, 1
    call .got_bit
    inc cx
    jmp short .do_copy

.get_elias:
    mov cl, 1

.elias_loop:
    add al, al
    jnz .got_bit
    lodsb
    adc al, al

.got_bit:
    jc .done

.elias_bt:
    add al, al
    adc cx, cx
    jmp short .elias_loop

.e8_loop:
    scasb
    jne .next
    add [di], cx

.next:
    loop .e8_loop
    ror ah, 1
    xchg ax, cx

.eof:
    inc ch
    jz .done

.mp_loop:
    lodsw
    xchg ax, cx
    jcxz .done
    sub di, cx
    inc ax
    js .e8_loop
    lodsb
    mov bl, al
    mov bh, 0xff

.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    loop .floop
    jmp short .mp_loop

.done:
    ret
