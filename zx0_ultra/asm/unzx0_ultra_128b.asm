;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_ultra_128b.asm)
; Description: 128-byte Flagship Multi-Modal 2D x86-16 decompressor (exact 128B budget)
;              unifying Sentinel-driven Variable-Bit Literals, Chained x86 E8/E9 rel16 CALL/JMP,
;              and Unified 16-Bit Word / 8-Bit Byte Multi-Pass 2D Stride-Delta integration:
;              - Sentinel Variable-Bit Literals (4..8 bits/literal): at entry, reads a 16-bit
;                header word into DX initializing DL with the first 7 bitstream bits + sentinel
;                bit 0 ('1'), and DH with literal width sentinel 1 << (8 - lit_bits) (0x01=8b,
;                0x02=7b ASCII, 0x04=6b VGA DAC). Unpacks variable-bit literals via stosb.
;              - Algebraic Multi-Mode EOF Dispatch & Fall-Through E8/E9 Post-Filter:
;                - Mode 0 (V=256): Standard 0-byte trailer EOF (inc ch -> 0x00, jz .done).
;                - Mode 1 (V=257): Enters .mp_loop directly in .floop (inc ax -> 0x0000, SF=0).
;                - Mode 2 (V=1049): Runs 1 pass of .e8_loop (AL=0xE8, CALL rel16), then transitions
;                  to .floop.
;                - Mode 3 (V=537): Runs Pass 1 (AL=0xE8, CALL rel16) and Pass 2 (AL=0xE9,
;                  JMP rel16), then transitions automatically to .floop on Pass 3+.
;              - Compact 4-Byte-Per-Pass 16-Bit Word / 8-Bit Byte 2D Stride Integrator:
;                Reads count_u16 + packed mode_stride_u16 (AL = 256 - stride, AH = 0x80 for
;                16-bit Word-mode or 0x00 for 8-bit Byte-mode). Inside .floop, 'add [di], al'
;                computes the low-byte sum and sets CF; 'inc di / inc dx / jns .fnext' preserves
;                CF and branches on SF of DX; in 16-bit Word-mode, 'adc [di], ah / inc di / dec cx'
;                propagates the carry into the high byte (saving 2 bytes per pass in the stream).
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Flagship 128B Multi-Modal 2D + Variable-Bit size-coding: Oleksandr Dreval (MIT License)
;
; License: MIT (SPDX-License-Identifier: MIT)
;--------------------------------------------------------------------------------------------------

bits 16

dzx0:
    lodsw
    xchg ax, dx
    dec bx

.literals:
    call .get_elias

.lit_loop:
    mov al, dh

.lit_bit:
    call .get_bit
    adc al, al
    jnc .lit_bit
    stosb
    loop .lit_loop
    call .get_bit
    jc .get_offset
    call .get_elias

.do_copy:
    push si
    lea si, [di+bx]
    rep movsb
    pop si
    call .get_bit
    jnc .literals

.get_offset:
    mov cl, 0xfe
    call .elias_loop
    inc cx
    js .eof

    mov ah, cl
    lodsb
    xchg ax, bx
    mov cx, 1
    rcr bx, 1
    call .got_bit
    inc cx
    jmp short .do_copy

.get_elias:
    mov cl, 1

.elias_loop:
    call .get_bit

.got_bit:
    jc .done
    call .get_bit
    adc cx, cx
    jmp short .elias_loop

.get_bit:
    add dl, dl
    jnz .done
    mov dl, [si]
    inc si
    adc dl, dl

.done:
    ret

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
    lodsw
    mov dh, ah
    cbw
    xchg ax, bx

.floop:
    mov ax, [bx+di]
    add [di], al
    inc di
    inc dx
    jns .fnext
    adc [di], ah
    inc di
    dec cx

.fnext:
    loop .floop
    jmp short .mp_loop
