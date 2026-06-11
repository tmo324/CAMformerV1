module a4_9bit_adc #(
    parameter WIDTH = 9  // 9-bit ADC
)(
    input  logic                clk,     // Clock input
    input  logic                rst_n,   // Active low reset
    input  logic                enable,  // Enable ADC operation
    input  logic [WIDTH-1:0]    vin,    // Input voltage (digital representation)
    output logic [WIDTH-1:0]    dout,   // Digital output
    output logic                valid    // Output valid signal
);

    // Internal registers
    logic [WIDTH-1:0] sample_reg;
    logic [1:0] state;

    // Simple state machine for ADC operation
    localparam IDLE    = 2'b00;
    localparam SAMPLE  = 2'b01;
    localparam CONVERT = 2'b10;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            dout <= '0;
            valid <= 1'b0;
            sample_reg <= '0;
        end
        else begin
            case (state)
                IDLE: begin
                    valid <= 1'b0;
                    if (enable) begin
                        state <= SAMPLE;
                    end
                end

                SAMPLE: begin
                    sample_reg <= vin;
                    state <= CONVERT;
                end

                CONVERT: begin
                    dout <= sample_reg;
                    valid <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule