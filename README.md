# GPU-Registerfile-Virtualization

This is a GPU register file virtualization code that was developed based on GPGPU-Sim for

Hyeran Jeon, Gokul Subramanian Ravi, Nam Sung Kim, and Murali Annavaram, ¡°GPGPU Register File Virtualization,¡± The 48th IEEE/ACM International Symposium on Microarchitecture (MICRO), Waikiki, Hawaii, December 2015


- Compilation & Execution: The same with GPGPU-Sim

- Sample configuration: sample_gpgpusim.config

- How can I turn off register file virtualization? : By commenting out -DRENAMING in the Makefiles.

- Expected simulation outputs: 
You can check the register file virtualization statistics under the following keywords in the simulation output file:

total physical register allocated = 2672127    // total number of physical registers used
total arch register allocated = 5301100          // total number of architected registers used
max physical register allocated = 247           // peak number of physical registers allocated
max arch register allocated = 288                 // peak number of architected registers allocated
total phys subarray utils  = 21889                 // physical register subarray utilization
total arch subarray utils  = 33366                  // architectural register subarray utilization

gpu_reg_rename_table_access = 851056      // total number of accesses to renaming table

- My apologies but I am not currently actively maintaining the code. Please cite the above paper when you write a paper by using this code.