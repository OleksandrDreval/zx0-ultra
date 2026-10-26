;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0
; Description: ZX0 v2 (non-classic) decompression routine optimized for 16-bit real mode in
;              MS-DOS .COM executables where segment registers DS = ES = SS = CS.
;              Size: 67 bytes (versus 81 bytes in the reference unzx0_8088.S port).
;
; Credits & Copyright:
;   - Algorithm & Format: Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 Architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Implementation & Size-Coding (67 bytes): Oleksandr Dreval (MIT License)
;
; License: MIT (SPDX-License-Identifier: MIT)
;
; Input Registers:
;   - DS:SI: Pointer to the start of the compressed ZX0 byte stream in source memory.
;   - ES:DI: Pointer to the destination buffer for decompressed output.
;   (Note: In standard MS-DOS .COM files, DS = ES = CS, rendering segment prefixes redundant).
;
; Output Registers:
;   - DI: Points to the byte immediately AFTER the last decompressed output byte.
;   - SI: Points to the byte immediately AFTER the end of the compressed input stream.
;
; Side Effects:
;   - Clobbers AX, BX, CX, SI, and DI registers during bit stream processing and block copies.
;   - The decompressor DOES NOT preserve General Purpose Registers; callers must preserve any
;     needed state prior to invocation to minimize decompressor code footprint.
;
; Architectural & Micro-Optimizations relative to reference unzx0_8088.S (-14 bytes net reduction):
;
;   [1] ELIMINATED push es (1b) + pop ds (1b) within copy_match block = -2 bytes saving.
;       The reference design switched DS to match ES so 'rep movsb' could copy historical data
;       from the output buffer. In .COM architecture, DS=ES inherently; these instructions were
;       redundant overhead.
;
;   [2] REPLACED push ds (1b) + pop ds (1b) with push si (1b) + pop si (1b) = 0 byte diff, safer logic.
;       The original code preserved DS across copies via the stack. We instead preserve SI (the
;       compressed stream reader pointer directly), avoiding unnecessary segment mutations.
;
;   [3] OPTIMIZED BX register usage for match offset:
;       Replaced DX with BX to utilize 16-bit effective addressing 'lea si, [di+bx]' (3b) instead of
;       'mov si, di' (2b) + 'add si, dx' (2b) = -1 byte saving.
;
;   [4] ELIMINATED redundant stc (1b) before rcr bx, 1 in .get_offset = -1 byte saving.
;       Prior to reaching rcr, .elias_loop terminates exclusively via 'jc .done' (requiring CF=1).
;       Since inc cl, jz, mov, and inc si preserve CF in x86 architecture, CF is guaranteed 1.
;
;   [5] REPLACED mov cx, 1 (3b) with mov cl, 1 (2b) in .get_elias = -1 byte saving.
;       In standard 16-bit .COM executables, initial CH is zero (0x00) upon startup (CX=0x00FF),
;       and all subsequent entries to .get_elias within the loop occur immediately after 'rep movsb'
;       which sets CX=0x0000. Therefore, CH is guaranteed zero and an 8-bit immediate load suffices.
;
;   [6] ELIMINATED cld (1b) instruction = -1 byte saving.
;       MS-DOS .COM environment conventions guarantee Direction Flag is 0 (DF=0) upon entry and
;       across interrupt dispatches.
;
;   [7] OPTIMIZED BX initialization:
;       Replaced 'mov bx, -1' (3b) with 'dec bx' (1b) relying on standard DOS .COM startup state
;       where BX is zeroed by the loader (BX=0x0000 -> dec bx yields 0xFFFF = -1) = -2 bytes saving.
;
;   [8] MERGED backtrack branch into .got_bit:
;       Replaced 'jc .got_offs / call .elias_bt / .got_offs:' (5b) with 'call .got_bit' (3b) = -2 bytes saving.
;       Near CALL preserves CF; .got_bit immediately executes 'jc .done' (ret) when CF=1 or falls into .elias_bt.
;
;   [9] REPLACED inc cl / jz (4b) with inc cx / js (3b) in .get_offset = -1 byte saving.
;       CX is guaranteed 0x0000 before .get_offset (from rep movsb). Seeding CL=0xFE gives CX=0x00FE.
;       Valid offsets (1..255, 0..7 shifts) yield CX <= 0x7F7F -> inc cx yields <= 0x7F80 (SF=0, CF=1).
;       EOF marker (256, 8 shifts) yields CX = 0xFEFF -> inc cx yields 0xFF00 (SF=1).
;--------------------------------------------------------------------------------------------------


bits 16


dzx0:
    mov al, 0x80            ; Initialize bit queue with a single marker '1' in MSB; when shifted out, triggers lodsb
    dec bx                  ; Initial match offset = -1 (0xFFFF). Relies on DOS .COM startup state (BX=0 -> dec bx=0xFFFF)


;--------------------------------------------------------------------------------------------------
; Routine Name: .literals
; Description: Unpacks literal byte sequences directly from the compressed stream to destination buffer.
;              Block format: [implicit ctrl=0] [Elias(count)] [byte_1]...[byte_N].
;              The initial stream block is ALWAYS literal; its initial control bit is assumed 0.
; Input Registers:
;   - AL: Current bit queue state.
;   - DS:SI: Source pointer to compressed byte stream.
;   - ES:DI: Destination pointer to decompression output buffer.
; Output Registers:
;   - AL: Updated bit queue state.
;   - SI: Advanced past consumed literal stream bytes.
;   - DI: Advanced past decompressed output bytes.
; Side Effects: Transfers literal bytes from DS:SI to ES:DI; clobbers CX during transfer.
;--------------------------------------------------------------------------------------------------
.literals:
    call .get_elias         ; Decode literal run length into CX via Elias gamma decoding
    rep movsb               ; Copy CX literal bytes from DS:SI to ES:DI (in .COM mode, DS and ES are unified)

    add al, al              ; Shift bit queue left; MSB moves into CF to determine next block type
                            ; CF=0: transition to copy-last-offset block (reuse previous match offset in BX)
                            ; CF=1: transition to copy-new-offset block (decode new offset from stream)
    jc .get_offset          ; If CF=1, branch to fetch a new compression offset from the stream

    call .get_elias         ; If CF=0, read repeat copy run length utilizing existing negative offset in BX


;--------------------------------------------------------------------------------------------------
; Routine Name: .do_copy
; Description: Executes dictionary historical copying from previously decompressed output buffer space.
; Input Registers:
;   - CX: Number of bytes to replicate from window.
;   - BX: Negative 16-bit offset distance relative to current DI writing head.
;   - AL: Current bit queue state.
;   - DS:SI: Compressed stream reader pointer (preserved across copy).
;   - ES:DI: Destination buffer writing head pointer.
; Output Registers:
;   - DI: Advanced by CX bytes.
;   - CX: Zeroed after block transfer via rep movsb.
;   - AL: Updated bit queue state after control bit shift.
;   - SI: Preserved compressed stream reader pointer.
; Side Effects: Copies historical bytes to ES:DI; preserves SI via stack push/pop; BX offset remains intact.
;--------------------------------------------------------------------------------------------------
.do_copy:
    push si                 ; Preserve SI (compressed stream reader pointer) across historical data repetition
    lea si, [di+bx]         ; Calculate historical match source address (SI = DI + BX) in 3 bytes
    rep movsb               ; Copy CX bytes from match history ES:SI -> ES:DI (handles overlapping runs naturally)
    pop si                  ; Restore SI to resume reading compressed token stream

    add al, al              ; Shift out next control bit from queue AL into CF
    jnc .literals           ; If CF=0, next sequence consists of raw literals; otherwise continue to .get_offset


;--------------------------------------------------------------------------------------------------
; Routine Name: .get_offset
; Description: Decodes a new historical window match offset and corresponding match run length.
;              Format: [ctrl=1] [Elias(msb+1)] [lsb_byte] [Elias(len-1)].
; Input Registers:
;   - AL: Current bit queue state.
;   - DS:SI: Pointer to compressed byte stream.
;   - ES:DI: Destination buffer pointer.
; Output Registers:
;   - BX: Updated negative 16-bit window offset.
;   - CX: Decoded match run length ready for .do_copy.
;   - AL: Updated bit queue state.
;   - SI: Advanced past offset and length stream bytes.
; Side Effects: Advances SI; branches directly to .do_copy; terminates execution (jz .done) if EOF marker is detected.
;--------------------------------------------------------------------------------------------------
.get_offset:
    mov cl, 0xfe            ; Seed CL with 0xFE (-2) to prepare Elias accumulator for negative MSB derivation
                            ; Upon accumulator loop completion and 'inc cx', CL becomes -(msb), forming BH
    call .elias_loop        ; Read Elias(msb+1) into CX, accumulating directly onto seed 0x00FE
    inc cx                  ; 1-byte increment: valid offsets stay <= 0x7F80 (SF=0); EOF (0xFEFF) becomes 0xFF00 (SF=1)
%ifndef DZX0_EOF_TARGET
%define DZX0_EOF_TARGET .done
%endif
    js DZX0_EOF_TARGET      ; If SF=1, end-of-stream EOF marker reached; branch to target immediately

    mov bh, cl              ; Store negative high-byte of offset into BH
    mov cx, 1               ; Initialize CX=1 as starting seed for match length Elias gamma decoding
    mov bl, [si]            ; Load LSB of match offset directly from compressed stream at [si]
    inc si                  ; Advance stream pointer SI past the consumed LSB byte
                            ; Note: BL contains inverted 7-bit LSB offset; bit 0 holds first length backtrack bit
                            ; CF is architecturally guaranteed to be 1 here from .elias_loop exit (jc .done)
    rcr bx, 1               ; Rotate BX right through CF: inserts sign-extending 1 into bit 15 while
                            ; shifting bit 0 of BL into CF as the initial backtrack length control bit
    call .got_bit           ; Near CALL preserves CF: if CF=1, 'jc .done' returns immediately; else runs .elias_bt

    inc cx                  ; Increment decoded length by +1 (recompense for Elias(length-1) encoding scheme)
    jmp short .do_copy      ; Proceed immediately to copy historical bytes using updated BX offset and CX length


;--------------------------------------------------------------------------------------------------
; Routine Name: .get_elias
; Description: Reads an integer encoded via interleaved Elias gamma representation with base seed CX=1.
;              Elias gamma encoding structure for integer N consists of N-1 leading control zeros
;              interleaved with data bits, terminated by a set bit (=1).
; Input Registers:
;   - AL: Active bit queue state.
;   - DS:SI: Pointer to compressed byte stream.
; Output Registers:
;   - CX: Decoded positive integer value (>= 1).
;   - AL: Updated bit queue state.
;   - SI: Advanced if fresh bytes were reloaded via lodsb.
; Side Effects: Sets CL=1 (CH is guaranteed 0); consumes stream bits from AL/SI.
;--------------------------------------------------------------------------------------------------
.get_elias:
    mov cl, 1               ; Initialize Elias accumulator seed to 1 (CH guaranteed 0 from startup / rep movsb)


;--------------------------------------------------------------------------------------------------
; Routine Name: .elias_loop / .elias_bt
; Description: Internal evaluation loop for Elias gamma parsing utilizing carry-flag shifting.
;              Mirrors Z80 bit queue optimization ('add a,a' / 'rla' equivalents in x86).
; Input Registers:
;   - CX: Seed value (1 for general lengths, 0x00FE for offset MSB decoding).
;   - AL: Bit queue state (value 0x80 indicates exhausted queue needing reload from SI).
;   - DS:SI: Pointer to compressed byte stream.
; Output Registers:
;   - CX: Accumulated result integer (or offset MSB).
;   - AL: Updated bit queue state.
;   - SI: Advanced if lodsb loaded a new byte.
; Side Effects: Advances SI via lodsb whenever AL queue exhausts to zero; sets CF=1 on exit (jc .done).
;--------------------------------------------------------------------------------------------------
.elias_loop:
    add al, al              ; Shift bit queue left; next control bit transitions into Carry Flag (CF)
    jnz .got_bit            ; If AL is non-zero after shift, bits remain in queue; skip memory fetch
    lodsb                   ; Queue exhausted (AL=0): load fresh 8-bit block from stream into AL (SI++)
    adc al, al              ; Shift newly loaded byte left, incorporating previous carry flag into bit 0

.got_bit:
    jc .done                ; If CF=1 (stop bit encountered), Elias integer decoding is complete
                            ; If CF=0, read subsequent interleaved payload bit from queue

.elias_bt:
    add al, al              ; Shift out interleaved data bit from AL directly into Carry Flag (CF)
    adc cx, cx              ; Double accumulator CX and integrate data bit from CF (CX = CX * 2 + CF)
    jmp short .elias_loop   ; Loop until stop bit finishes Elias gamma sequence

.done:
    ret                     ; Return to caller with decoded value in CX (or EOF exit when branching from .get_offset)