;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_adpk_77b.asm)
; Description: 77-byte Adaptive-k Offset ZX0-Ultra decompressor (+7 bytes vs 70-byte baseline,
;              -1 byte vs 78-byte unzx0_adpk_78b.asm).
;              Reads a 1-byte stream header specifying k in 1..12 offset LSB bits per NEW_OFFSET,
;              exploiting AH=0x00 at standard MS-DOS .COM entry so 'lodsb / xchg ax, bp' loads
;              BP = k in 2 bytes without requiring 'cbw'.
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Adaptive-k format & size-coding (77B): Oleksandr Dreval (MIT License)
;
; License: MIT (SPDX-License-Identifier: MIT)
;--------------------------------------------------------------------------------------------------

bits 16

dzx0:
    lodsb                   ; Read stream header k (1..12 offset LSB bits; AH=0x00 at .COM entry)
    xchg ax, bp             ; Preserve k in BP across the entire decompression (AX = old BP)
    mov al, 0x80
    dec bx

.literals:
    call .get_elias
    rep movsb
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
    dec cx                  ; CX was 0 -> 0xFFFF
    dec cx                  ; CX = 0xFFFE
    call .elias_loop
    inc cl
%ifndef DZX0_EOF_TARGET
%define DZX0_EOF_TARGET .ret
%endif
    jz DZX0_EOF_TARGET

    mov bx, cx              ; BX = -msb (16-bit negative, since CH = 0xFF)
    mov cx, bp              ; Loop k times to shift inverted LSB bits into BX
.off_lsb:
    call .get_bit
    adc bx, bx
    loop .off_lsb
    call .get_elias
    inc cx
    jmp short .do_copy

.get_elias:
    mov cl, 1

.elias_loop:
    call .get_bit
    jc .ret
    call .get_bit
    adc cx, cx
    jmp short .elias_loop

.get_bit:
    add al, al
    jnz .ret
    lodsb
    adc al, al

.ret:
    ret
