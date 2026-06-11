module a16_address_controller (
    input  logic                clk,
    input  logic                rst_n,
    input  logic [239:0]        indices_in,     // 30*8-1:0
    input  logic                indices_valid,
    input  logic                sram_ready,
    output logic [5:0]          chunk_id,
    output logic [29:0]         indices_mask,    // 30-1:0
    output logic                load_request,
    output logic                processing_done
);
    // States for controller FSM
    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        ANALYZE = 2'b01,
        REQUEST = 2'b10,
        WAIT    = 2'b11
    } state_t;

    // Internal registers
    state_t         current_state;
    state_t         next_state;
    logic [3:0]     chunk_counter;
    logic [29:0]    pending_indices;

    // Calculate which indices belong to current chunk
    always_comb begin
        for (int i = 0; i < 30; i++) begin
            logic [7:0] current_index;
            current_index = indices_in[i*8 +: 8];
            indices_mask[i] = (current_index[7:6] == chunk_id[1:0]);
        end
    end

    // State machine sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state   <= IDLE;
            chunk_counter   <= 4'h0;
            pending_indices <= 30'h0;
            chunk_id       <= 6'h0;
            load_request   <= 1'b0;
            processing_done <= 1'b0;
        end else begin
            current_state <= next_state;
            processing_done <= 1'b0;  // Default value
            
            case (current_state)
                ANALYZE: begin
                    if (indices_valid) begin
                        pending_indices <= {30{1'b1}};
                    end
                end
                
                REQUEST: begin
                    if (sram_ready) begin
                        chunk_counter <= chunk_counter + 4'h1;
                        chunk_id     <= {4'h0, chunk_counter[1:0]};
                        load_request <= 1'b1;
                    end
                end
                
                WAIT: begin
                    load_request <= 1'b0;
                    pending_indices <= pending_indices & ~indices_mask;
                    if (pending_indices == 30'h0) begin
                        processing_done <= 1'b1;
                    end
                end
                
                default: begin  // IDLE
                    chunk_counter   <= 4'h0;
                    load_request   <= 1'b0;
                end
            endcase
        end
    end

    // State machine combinational logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (indices_valid) begin
                    next_state = ANALYZE;
                end
            end
            
            ANALYZE: begin
                next_state = REQUEST;
            end
            
            REQUEST: begin
                if (sram_ready) begin
                    next_state = WAIT;
                end
            end
            
            WAIT: begin
                if (pending_indices == 30'h0) begin
                    next_state = IDLE;
                end else if (sram_ready) begin
                    next_state = REQUEST;
                end
            end
        endcase
    end

endmodule