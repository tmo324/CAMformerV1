module a12_sparseMM #(
    parameter ROWS = 30,
    parameter COLS = 1024,
    parameter BF16_WIDTH = 16,
    parameter INPUT_WIDTH = 8,
    parameter PARALLEL_COLS = 4  // Process 4 columns in parallel
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                start,
    input  logic [ROWS-1:0][INPUT_WIDTH-1:0]    vec_in,      // 30x8b input vector
    input  logic [ROWS-1:0][PARALLEL_COLS-1:0][BF16_WIDTH-1:0] matrix_cols, // 4 columns at once
    input  logic                                cols_valid,
    output logic [COLS-1:0][BF16_WIDTH-1:0]     result_out,  // 1024x1 BF16 output
    output logic                                done
);

    // State machine definition
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        PROCESS = 2'b01,
        OUTPUT = 2'b10
    } state_t;
    
    // State and control registers
    state_t state, next_state;
    logic [7:0] col_counter;  // Counter for 256 groups (1024/4)
    
    // Pipeline registers
    logic [ROWS-1:0][PARALLEL_COLS-1:0][BF16_WIDTH-1:0] mult_results_stage1;
    logic [PARALLEL_COLS-1:0][BF16_WIDTH-1:0] sum_stage1 [ROWS/2-1:0];  // First level adder tree
    logic [PARALLEL_COLS-1:0][BF16_WIDTH-1:0] sum_stage2 [ROWS/4-1:0];  // Second level adder tree
    logic [PARALLEL_COLS-1:0][BF16_WIDTH-1:0] final_sums;  // Final sums for 4 columns

    // Vector input register
    logic [ROWS-1:0][INPUT_WIDTH-1:0] vec_reg;

    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            col_counter <= '0;
            done <= 1'b0;
            vec_reg <= '0;
            for (int i = 0; i < COLS; i++) begin
                result_out[i] <= '0;
            end
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        col_counter <= '0;
                        done <= 1'b0;
                        vec_reg <= vec_in;  // Register input vector
                    end
                end

                PROCESS: begin
                    if (cols_valid) begin
                        // Pipeline Stage 1: Parallel Multiplications
                        for (int r = 0; r < ROWS; r++) begin
                            for (int c = 0; c < PARALLEL_COLS; c++) begin
                                mult_results_stage1[r][c] <= matrix_cols[r][c] * 
                                    {{(BF16_WIDTH-INPUT_WIDTH){1'b0}}, vec_reg[r]};
                            end
                        end

                        // Pipeline Stage 2: First level of adder tree
                        for (int c = 0; c < PARALLEL_COLS; c++) begin
                            for (int i = 0; i < ROWS/2; i++) begin
                                sum_stage1[i][c] <= mult_results_stage1[i*2][c] + 
                                                  mult_results_stage1[i*2+1][c];
                            end
                        end

                        // Pipeline Stage 3: Second level of adder tree
                        for (int c = 0; c < PARALLEL_COLS; c++) begin
                            for (int i = 0; i < ROWS/4; i++) begin
                                sum_stage2[i][c] <= sum_stage1[i*2][c] + 
                                                  sum_stage1[i*2+1][c];
                            end
                        end

                        // Pipeline Stage 4: Final accumulation
                        for (int c = 0; c < PARALLEL_COLS; c++) begin
                            final_sums[c] <= sum_stage2[0][c] + sum_stage2[1][c];
                        end

                        // Store results for 4 columns
                        for (int c = 0; c < PARALLEL_COLS; c++) begin
                            result_out[col_counter*PARALLEL_COLS + c] <= final_sums[c];
                        end

                        // Increment column counter
                        if (col_counter == (COLS/PARALLEL_COLS)-1) begin
                            col_counter <= '0;
                        end else begin
                            col_counter <= col_counter + 1;
                        end
                    end
                end

                OUTPUT: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: 
                if (start) next_state = PROCESS;
            
            PROCESS: begin
                if (cols_valid && col_counter == (COLS/PARALLEL_COLS)-1)
                    next_state = OUTPUT;
            end
            
            OUTPUT:
                next_state = IDLE;
            
            default:
                next_state = IDLE;
        endcase
    end

endmodule