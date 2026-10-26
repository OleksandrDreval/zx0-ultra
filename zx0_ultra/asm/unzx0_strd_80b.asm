;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_strd_80b.asm)
; Description: 80-byte ZX0-Ultra decompressor (+10 bytes vs 70-byte baseline) with in-place
;              arbitrary 16-bit Stride-Delta post-filter (out[i] += out[i - stride]).
;              Reuses final negative match offset BX = -stride directly at EOF, requiring zero
;              extra stack pushes and preserving final output pointer DI = output_end.
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - 16-bit Stride-Delta post-filter & size-coding (80B): Oleksandr Dreval (MIT License)
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

.eof:
    lodsw                   ; Read 16-bit filtered tail count (0 = unfiltered raw ZX0)
    xchg ax, cx
    jcxz .done
    sub di, cx              ; Rewind DI to (output_end - count); loop restores DI = output_end
.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    loop .floop

.done:
    ret
