# variable setup
set top_module "multiplier8"
set src "$top_module.v"
set res_dir "and"
#set top_module "i${numInput}_v$value"

set link_library []
set target_library [list /opt2/synopsys/SAED/SAED32_EDK/lib/stdcell_lvt/db_nldm/saed32lvt_tt1p05v25c.db]

#set designs [sh cat designs.lst]
#puts "$values"
set delays {5}
#set muls {and  nand  and_radix4  nand_radix4  benc_radix4  benc_radix8 benc_radix4_mux benc_radix8_mux}
set mul_options {{ -booth_encoding false -mult_radix4 false -mult_nand_based true}}
set adders {ling_adder hybrid_adder cond_sum_adder sklansky_adder brent_kung_adder bounded_fanout_adder}
foreach a $adders {
  
    foreach delay $delays {
        #set src "$v.v"
        read_file -format verilog ${src}
        #analyze  -format verilog designs
        #elaborate ${top_module}
        current_design ${top_module}
        link
        check_design
        set run_id [format "%s%s" $top_module $a ]
        puts "running $src..."
        set_max_delay $delay -from [all_inputs]
        set_dp_smartgen_options -all_options default
        if {$a != "auto_adder"} {
            set_dp_smartgen_options -booth_encoding false -mult_radix4 false -mult_nand_based false
            set_dp_smartgen_options -$a true
        }
        set compile_ultra_ungroup_dw false
        compile_ultra -gate_clock -no_autoungroup
        report_qor >> "$res_dir/report/hier_$run_id.rpt"
        report_resources -hierarchy -nosplit >> "$res_dir/report/hier_$run_id.rpt"
        report_timing >> "$res_dir/report/hier_$run_id.rpt"
        change_names -rules verilog -hierarchy -verbose
        write -hierarchy -f verilog -output "$res_dir/implementation/hier_${run_id}_d${delay}.v"
        ungroup -all -flatten
        check_design
        report_qor >> "$res_dir/report/$run_id.rpt"
        report_resources -hierarchy -nosplit >> "$res_dir/report/$run_id.rpt"
        report_timing >> "$res_dir/report/$run_id.rpt"
        change_names -rules verilog -hierarchy -verbose
        write -hierarchy -f verilog -output "$res_dir/implementation/${run_id}_d${delay}.v"
    }
  
}
quit

exit