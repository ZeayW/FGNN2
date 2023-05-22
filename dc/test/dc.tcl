# variable setup
set rtl "RocketCore"
set rtl_src "$rtl.v"
set top_module "Rocket"

set link_library []
set target_library [list /opt2/synopsys/SAED/SAED32_EDK/lib/stdcell_lvt/db_nldm/saed32lvt_tt1p05v25c.db]


set delays {10.0}
set adders {auto_adder}
foreach d $delays {
    foreach a $adders {
        read_file -format verilog ${rtl_src}
        current_design ${top_module}
        link
        check_design
        set run_id [format "%s_d%.2f_%s" $rtl $d $a]
        puts "running $run_id..."
        set_max_delay $d -from [all_inputs]
        set_dp_smartgen_options -all_options default
        if {$a != "auto_adder"} {
            set_dp_smartgen_options -$a true
        }
        set compile_ultra_ungroup_dw false
        compile_ultra -gate_clock -no_autoungroup
        report_qor >> "report/hier_$run_id.rpt"
        report_resources -hierarchy -nosplit >> "report/hier_$run_id.rpt"
        report_timing >> "report/hier_$run_id.rpt"
        change_names -rules verilog -hierarchy -verbose
        write -hierarchy -f verilog -output "implementation/hier_$run_id.v"
        ungroup -all -flatten
        check_design
        report_qor >> "report/$run_id.rpt"
        report_resources -hierarchy -nosplit >> "report/$run_id.rpt"
        report_timing >> "report/$run_id.rpt"
        change_names -rules verilog -hierarchy -verbose
        write -hierarchy -f verilog -output "implementation/$run_id.v"
    }
}

quit





