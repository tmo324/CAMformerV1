module a8_top3 #(
    parameter INPUT_COUNT = 32,    // Number of input elements
    parameter WIDTH = 14,          // Width of each element
    parameter CNT_WIDTH = $clog2(INPUT_COUNT)  // Counter width
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     start,           // Start processing
    input  logic [31:0][13:0]        data_in,        // 32 elements, 14 bits each
    output logic [2:0][13:0]         top3_out,       // Top 3 values output
    output logic                     done            // Processing complete
);

    // State encoding
    typedef enum logic [2:0] {
        IDLE,
        LOAD,
        START_PROC,
        RUN,
        OUTPUT
    } state_t;

    state_t state;

    // FIFO signals
    logic [WIDTH-1:0]    fifo_l_data, fifo_r_data;
    logic                fifo_l_push, fifo_r_push;
    logic                fifo_l_pop, fifo_r_pop;
    logic                fifo_l_empty, fifo_r_empty;
    logic [CNT_WIDTH:0]  fifo_l_count, fifo_r_count;
    
    // Control signals
    logic [CNT_WIDTH-1:0] target;
    logic [CNT_WIDTH-1:0] num_eq_pivot;
    logic [WIDTH-1:0]     pivot;
    logic                 select;  // 0 for FIFO_L, 1 for FIFO_R
    logic [CNT_WIDTH-1:0] load_count;
    logic [1:0]          found_count;

    // FIFO instances
    fifo #(
        .WIDTH(WIDTH),
        .DEPTH(INPUT_COUNT)
    ) fifo_l (
        .clk(clk),
        .rst_n(rst_n),
        .push(fifo_l_push),
        .pop(fifo_l_pop),
        .data_in(data_in[load_count]),
        .data_out(fifo_l_data),
        .empty(fifo_l_empty),
        .count(fifo_l_count)
    );

    fifo #(
        .WIDTH(WIDTH),
        .DEPTH(INPUT_COUNT)
    ) fifo_r (
        .clk(clk),
        .rst_n(rst_n),
        .push(fifo_r_push),
        .pop(fifo_r_pop),
        .data_in(pivot),
        .data_out(fifo_r_data),
        .empty(fifo_r_empty),
        .count(fifo_r_count)
    );

    // Main state machine and control logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            target <= 3;  // We want top 3
            num_eq_pivot <= '0;
            pivot <= '0;
            select <= 1'b0;
            load_count <= '0;
            found_count <= '0;
            done <= 1'b0;
            fifo_l_push <= 1'b0;
            fifo_l_pop <= 1'b0;
            fifo_r_push <= 1'b0;
            fifo_r_pop <= 1'b0;
            top3_out <= '{default:'0};
        end else begin
            // Default values
            fifo_l_push <= 1'b0;
            fifo_l_pop <= 1'b0;
            fifo_r_push <= 1'b0;
            fifo_r_pop <= 1'b0;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        state <= LOAD;
                        load_count <= '0;
                        target <= 3;
                        num_eq_pivot <= '0;
                    end
                end

                LOAD: begin
                    if (load_count < INPUT_COUNT) begin
                        fifo_l_push <= 1'b1;
                        load_count <= load_count + 1;
                    end else begin
                        state <= START_PROC;
                    end
                end

                START_PROC: begin
                    if (fifo_r_count + num_eq_pivot <= target) begin
                        // Keep elements in right FIFO
                        target <= target - fifo_r_count - num_eq_pivot;
                        select <= 1'b0;
                        fifo_l_pop <= 1'b1;
                        state <= RUN;
                    end else if (fifo_r_count > target) begin
                        // Too many elements in right FIFO
                        select <= 1'b1;
                        fifo_r_pop <= 1'b1;
                        state <= RUN;
                    end else begin
                        // Found one of top 3
                        top3_out[found_count] <= pivot;
                        if (found_count == 2) begin
                            state <= OUTPUT;
                        end else begin
                            found_count <= found_count + 1;
                            num_eq_pivot <= '0;
                            state <= START_PROC;
                        end
                    end
                end

                RUN: begin
                    if (!select && !fifo_l_empty) begin
                        pivot <= fifo_l_data;
                        fifo_l_pop <= 1'b1;
                        state <= START_PROC;
                    end else if (select && !fifo_r_empty) begin
                        pivot <= fifo_r_data;
                        fifo_r_pop <= 1'b1;
                        state <= START_PROC;
                    end else begin
                        state <= START_PROC;
                    end
                end

                OUTPUT: begin
                    done <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule