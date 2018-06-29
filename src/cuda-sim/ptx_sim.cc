// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ptx_sim.h"
#include <string>
#include "ptx_ir.h"
#include "ptx.tab.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/shader.h"

void feature_not_implemented( const char *f );

std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;
unsigned long long g_ptx_cta_info_uid = 1;

ptx_cta_info::ptx_cta_info( unsigned sm_idx )
{
   assert( g_ptx_cta_info_sm_idx_used.find(sm_idx) == g_ptx_cta_info_sm_idx_used.end() );
   g_ptx_cta_info_sm_idx_used.insert(sm_idx);

   m_sm_idx = sm_idx;
   m_uid = g_ptx_cta_info_uid++;
}

void ptx_cta_info::add_thread( ptx_thread_info *thd )
{
   m_threads_in_cta.insert(thd);
}

unsigned ptx_cta_info::num_threads() const
{
   return m_threads_in_cta.size();
}

void ptx_cta_info::check_cta_thread_status_and_reset()
{
   bool fail = false;
   if ( m_threads_that_have_exited.size() != m_threads_in_cta.size() ) {
      printf("\n\n");
      printf("Execution error: Some threads still running in CTA during CTA reallocation! (1)\n");
      printf("   CTA uid = %Lu (sm_idx = %u) : %lu running out of %zu total\n", 
             m_uid, 
             m_sm_idx,
             (m_threads_in_cta.size() - m_threads_that_have_exited.size()), m_threads_in_cta.size() );
      printf("   These are the threads that are still running:\n");
      std::set<ptx_thread_info*>::iterator t_iter;
      for ( t_iter=m_threads_in_cta.begin(); t_iter != m_threads_in_cta.end(); ++t_iter ) {
         ptx_thread_info *t = *t_iter;
         if ( m_threads_that_have_exited.find(t) == m_threads_that_have_exited.end() ) {
            if ( m_dangling_pointers.find(t) != m_dangling_pointers.end() ) {
               printf("       <thread deleted>\n");
            } else {
               printf("       [done=%c] : ", (t->is_done()?'Y':'N') );
               t->print_insn( t->get_pc(), stdout );
               printf("\n");
            }
         }
      }
      printf("\n\n");
      fail = true;
   }
   if ( fail ) {
      abort();
   }

   bool fail2 = false;
   std::set<ptx_thread_info*>::iterator t_iter;
   for ( t_iter=m_threads_in_cta.begin(); t_iter != m_threads_in_cta.end(); ++t_iter ) {
      ptx_thread_info *t = *t_iter;
      if ( m_dangling_pointers.find(t) == m_dangling_pointers.end() ) {
         if ( !t->is_done() ) {
            if ( !fail2 ) {
               printf("Execution error: Some threads still running in CTA during CTA reallocation! (2)\n");
               printf("   CTA uid = %Lu (sm_idx = %u) :\n", m_uid, m_sm_idx );
               fail2 = true;
            }
            printf("       ");
            t->print_insn( t->get_pc(), stdout );
            printf("\n");
         }
      }
   }
   if ( fail2 ) {
      abort();
   }
   m_threads_in_cta.clear();
   m_threads_that_have_exited.clear();
   m_dangling_pointers.clear();
}

void ptx_cta_info::register_thread_exit( ptx_thread_info *thd )
{
   assert( m_threads_that_have_exited.find(thd) == m_threads_that_have_exited.end() );
   m_threads_that_have_exited.insert(thd);
}

void ptx_cta_info::register_deleted_thread( ptx_thread_info *thd )
{
   m_dangling_pointers.insert(thd);
}

unsigned ptx_cta_info::get_sm_idx() const
{
   return m_sm_idx;
}

unsigned g_ptx_thread_info_uid_next=1;
unsigned g_ptx_thread_info_delete_count=0;

ptx_thread_info::~ptx_thread_info()
{
   g_ptx_thread_info_delete_count++;
#ifdef RENAMING
   for(std::map<const char*, reg_track*>::iterator it = m_reg_stat_map.begin();
		it != m_reg_stat_map.end(); ++it)
		free(it->second);
   m_reg_stat_map.clear();
#endif
}

ptx_thread_info::ptx_thread_info( kernel_info_t &kernel )
    : m_kernel(kernel)
{
   m_uid = g_ptx_thread_info_uid_next++;
   m_core = NULL;
   m_barrier_num = -1;
   m_at_barrier = false;
   m_valid = false;
   m_gridid = 0;
   m_thread_done = false;
   m_cycle_done = 0;
   m_PC=0;
   m_icount = 0;
   m_last_effective_address = 0;
   m_last_memory_space = undefined_space; 
   m_branch_taken = 0;
   m_shared_mem = NULL;
   m_cta_info = NULL;
   m_local_mem = NULL;
   m_symbol_table = NULL;
   m_func_info = NULL;
   m_hw_tid = -1;
   m_hw_wid = -1;
   m_hw_sid = -1;
   m_last_dram_callback.function = NULL;
   m_last_dram_callback.instruction = NULL;
   m_regs.push_back( reg_map_t() );
   m_debug_trace_regs_modified.push_back( reg_map_t() );
   m_debug_trace_regs_read.push_back( reg_map_t() );
   m_callstack.push_back( stack_entry() );
   m_RPC = -1;
   m_RPC_updated = false;
   m_last_was_call = false;
   m_enable_debug_trace = false;
   m_local_mem_stack_pointer = 0;
   m_gpu = NULL;
   m_last_set_operand_value=ptx_reg_t();
}

const ptx_version &ptx_thread_info::get_ptx_version() const 
{ 
   return m_func_info->get_ptx_version(); 
}

#ifdef RENAMING
// 1st set. 0, 1, 32, 4
#define REGSTAT_PRINT(FN, CTA, W, T, ...)\
	if(get_hw_ctaid() == CTA && get_hw_wid() == W && get_hw_tid() == T && get_hw_sid() == 0)\
	{\
		FILE *fw = fopen(FN, "a");\
		fprintf(fw, __VA_ARGS__);\
		fclose(fw);\
	}

int ptx_thread_info::checkDeadRegCnt(const class inst_t *inst)
{
   addr_t pc = next_instr();
   //assert( pc == inst->pc ); // make sure timing model and functional model are in sync
   if(pc != inst->pc)
      return true;
   const ptx_instruction *pI = m_func_info->get_instruction(pc);
	
   return pI->m_dead_regs.size();
}
/*
bool ptx_thread_info::checkRegStat(const class inst_t *inst)
{
   addr_t pc = next_instr();
   //assert( pc == inst->pc ); // make sure timing model and functional model are in sync
   if(pc != inst->pc)
      return true;
   const ptx_instruction *pI = m_func_info->get_instruction(pc);

   bool ready = true;
   if(pI->m_operands.size()>1)
   {
       const operand_info &src1 = pI->src1();
       //INST_STREAM_PRINT("r1: %s ", src1.is_reg()?src1.get_symbol()->name().c_str():"");
       if(src1.is_reg()) ready = isRegReady(src1.get_symbol()->name().c_str(), pc);
   }
   if(pI->m_operands.size()>2 && ready)
   {
       const operand_info &src2 = pI->src2();
       //INST_STREAM_PRINT("r2: %s ", src2.is_reg()?src2.get_symbol()->name().c_str():"");
       if(src2.is_reg()) ready = isRegReady(src2.get_symbol()->name().c_str(), pc);
   }
   if(pI->m_operands.size()>3 && ready)
   {
       const operand_info &src3 = pI->src3();
       //INST_STREAM_PRINT("r3: %s ", src3.is_reg()?src3.get_symbol()->name().c_str():"");
       if(src3.is_reg()) ready = isRegReady(src3.get_symbol()->name().c_str(), pc);
   }

   return ready;
}

bool ptx_thread_info::isRegReady(const char *name, addr_t pc)
{
   bool ready = true;
   if(searchRegStat(name))
   {
      std::map<const char*,reg_track*>::iterator it;
      it = m_reg_stat_map.find(name);
      reg_track *t = it->second;
      if((t->stat == dead))
      {
         ready = false;
      }

   }
   return ready;
}
*/
void ptx_thread_info::addRegStat(const char *name, bool isRead)
{  
	reg_track *t = (reg_track*)malloc(sizeof(reg_track));
	memset(t, 0, sizeof(reg_track));
	if(!isRead)
	{ 
		t->tot_accesses = 0;

		t->modified = true;
		t->last_write_time = gpu_sim_cycle;
		t->num_accesses = 0;
		t->last_op = reg_write;
		t->stat = live;
	}
	else
	{
		t->modified = false;
		t->last_read_time = gpu_sim_cycle;
		t->last_write_time = 0;
		t->last_op = reg_read;
		t->num_accesses = 1;
		t->stat = live;
	}
	t->first_access_time = gpu_sim_cycle;

	std::pair<std::map<const char*,reg_track*>::iterator,bool> ret;
	ret = m_reg_stat_map.insert(std::pair<const char*,reg_track*>(name, t));
	if(ret.second == false)
	{
	   printf("err: %s is already exist\n", name);
	}
}

bool ptx_thread_info::searchRegStat(const char *name)
{
   std::map<const char*,reg_track*>::iterator it;
   it = m_reg_stat_map.find(name);
   if(it == m_reg_stat_map.end()) 
		return false;
   return true;
}

void ptx_thread_info::checkDeadReg(const ptx_instruction *pI)
{
	if(pI->m_reg_update)
	{
		for(int i = 0; i < pI->m_dead_regs.size(); i++)
		{
			updateRegStat(pI->m_dead_regs[i], true, true, pI);
		}
	}
}
#if 0
void ptx_thread_info::RegStatFSM(reg_track *t, bool isRead, bool llo, bool isLLO, bool change)
{
	bool changed = change;
    if(change)
    {
		if(t->write_in_strand)// remove
		{
		 t->last_stat = t->stat;
         t->stat = dead;
		}
    }else{
			switch(t->stat)
			{
				case slr:
					if(llo){t->stat = llr;changed=true;}
					else t->stat = slr;
					break;
				case pot_llr:
					if(!llo)
					{
						if(isRead)
						{
							t->stat = llr;
							t->has_copy_in_strf = true;
						}
						else
							t->stat = slr;
						changed = true;
					}
					break;
				case llr:
					if(!llo){if(isRead)t->stat = llr;else{t->stat = slr;changed = true;}}
					break;
				case dead:
					if(!isRead && !llo){t->stat = slr; changed = true;}
					else if(isRead && !llo){t->stat = t->last_stat; changed = true; }
					break;
			}
    }

	if(changed)
	{
		last_print = gpu_sim_cycle;
		if(m_log_thread)
		{
			char fname[50];
			for(std::map<const char*,reg_track*>::iterator it=m_reg_stat_map.begin(); it!=m_reg_stat_map.end(); ++it)
			{
			   reg_track *rt = it->second;
			}
		}
	}
}
#endif

void ptx_thread_info::updateRegStat(const char *name, bool isRead, bool last_read, const ptx_instruction* pI)
{
	if(searchRegStat(name))
	{
		std::map<const char*,reg_track*>::iterator it;
		it = m_reg_stat_map.find(name);
		reg_track *t = it->second;

		//RegStatFSM(t, isRead, false, isLLO, last_read && isRead);

		if(isRead)
		{
			t->last_read_time = gpu_sim_cycle;
			t->last_op = reg_read;
			t->num_accesses++;
			if(last_read && isRead && 
				((pI->reconvergence_pc == ((address_type)-1)) || (pI->reconvergence_pc == ((address_type)-2))) )
			{
				((shader_core_ctx*)m_core)->renameReg(m_hw_ctaid, m_hw_wid, name, dead, t->num_accesses);
				t->stat = dead;
			}
		}
		else
		{
			t->last_write_time = gpu_sim_cycle;
			t->tot_accesses += t->num_accesses;
			t->num_accesses = 0;
			t->last_op = reg_write;
#ifndef MULT_RF
			((shader_core_ctx*)m_core)->renameReg(m_hw_ctaid, m_hw_wid, name, live, 0);
			t->stat = live;
#endif

		}
	}
	else
	{
		addRegStat(name, isRead);
		std::map<const char*,reg_track*>::iterator it;
		it = m_reg_stat_map.find(name);
		reg_track *t = it->second;
#ifndef MULT_RF
		((shader_core_ctx*)m_core)->renameReg(m_hw_ctaid, m_hw_wid, name, live, 0);
		t->stat = live;
			
#endif
   	}
}

#endif

void ptx_thread_info::set_done() 
{
   assert( !m_at_barrier );
   m_thread_done = true;
   m_cycle_done = gpu_sim_cycle; 
}

unsigned ptx_thread_info::get_builtin( int builtin_id, unsigned dim_mod ) 
{
   assert( m_valid );
   switch ((builtin_id&0xFFFF)) {
   case CLOCK_REG:
      return (unsigned)(gpu_sim_cycle + gpu_tot_sim_cycle);
   case CLOCK64_REG:
      abort(); // change return value to unsigned long long?
	  // GPGPUSim clock is 4 times slower - multiply by 4
	   return (gpu_sim_cycle + gpu_tot_sim_cycle)*4;
   case HALFCLOCK_ID:
      // GPGPUSim clock is 4 times slower - multiply by 4
	  // Hardware clock counter is incremented at half the shader clock frequency - divide by 2 (Henry '10)
      return (gpu_sim_cycle + gpu_tot_sim_cycle)*2;
   case CTAID_REG:
      assert( dim_mod < 3 );
      if( dim_mod == 0 ) return m_ctaid.x;
      if( dim_mod == 1 ) return m_ctaid.y;
      if( dim_mod == 2 ) return m_ctaid.z;
      abort();
      break;
   case ENVREG_REG:{
	int index = builtin_id >> 16;
	dim3 gdim = this->get_core()->get_kernel_info()->get_grid_dim();
		switch(index){
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			return 0;
			break;
		case 6:
			return gdim.x;
		case 7:
			return gdim.y;
		case 8:
			return gdim.z;
		case 9:
			if(gdim.z == 1 && gdim.y == 1)
				return 1;
			else if(gdim.z == 1)
				return 2;
			else
				return 3;
			break;
		default:
			break;
		}
   }
   case GRIDID_REG:
      return m_gridid;
   case LANEID_REG: feature_not_implemented( "%laneid" ); return 0;
   case LANEMASK_EQ_REG: feature_not_implemented( "%lanemask_eq" ); return 0;
   case LANEMASK_LE_REG: feature_not_implemented( "%lanemask_le" ); return 0;
   case LANEMASK_LT_REG: feature_not_implemented( "%lanemask_lt" ); return 0;
   case LANEMASK_GE_REG: feature_not_implemented( "%lanemask_ge" ); return 0;
   case LANEMASK_GT_REG: feature_not_implemented( "%lanemask_gt" ); return 0;
   case NCTAID_REG:
      assert( dim_mod < 3 );
      if( dim_mod == 0 ) return m_nctaid.x;
      if( dim_mod == 1 ) return m_nctaid.y;
      if( dim_mod == 2 ) return m_nctaid.z;
      abort();
      break;
   case NTID_REG:
      assert( dim_mod < 3 );
      if( dim_mod == 0 ) return m_ntid.x;
      if( dim_mod == 1 ) return m_ntid.y;
      if( dim_mod == 2 ) return m_ntid.z;
      abort();
      break;
   case NWARPID_REG: feature_not_implemented( "%nwarpid" ); return 0;
   case PM_REG: feature_not_implemented( "%pm" ); return 0;
   case SMID_REG: feature_not_implemented( "%smid" ); return 0;
   case TID_REG:
      assert( dim_mod < 3 );
      if( dim_mod == 0 ) return m_tid.x;
      if( dim_mod == 1 ) return m_tid.y;
      if( dim_mod == 2 ) return m_tid.z;
      abort();
      break;
   case WARPSZ_REG: feature_not_implemented( "WARP_SZ" ); return 0;
   default:
      assert(0);
   }
   return 0;
}

void ptx_thread_info::set_info( function_info *func ) 
{
  m_symbol_table = func->get_symtab();
  m_func_info = func;
  m_PC = func->get_start_PC();
}

void ptx_thread_info::cpy_tid_to_reg( dim3 tid )
{
   //copies %tid.x, %tid.y and %tid.z into $r0
   ptx_reg_t data;
   data.s64=0;

   data.u32=(tid.x + (tid.y<<16) + (tid.z<<26));

   const symbol *r0 = m_symbol_table->lookup("$r0");
   if (r0){
	   //No need to set pid if kernel doesn't use it
	   set_reg(r0,data);
   }
}

void ptx_thread_info::print_insn( unsigned pc, FILE * fp ) const
{
   m_func_info->print_insn(pc,fp);
}

static void print_reg( FILE *fp, std::string name, ptx_reg_t value, symbol_table *symtab )
{
   const symbol *sym = symtab->lookup(name.c_str());
   fprintf(fp,"  %8s   ", name.c_str() );
   if( sym == NULL ) {
      fprintf(fp,"<unknown type> 0x%llx\n", (unsigned long long ) value.u64 );
      return;
   }
   const type_info *t = sym->type();
   if( t == NULL ) {
      fprintf(fp,"<unknown type> 0x%llx\n", (unsigned long long ) value.u64 );
      return;
   }
   type_info_key ti = t->get_key();

   switch ( ti.scalar_type() ) {
   case S8_TYPE:  fprintf(fp,".s8  %d\n", value.s8 );  break;
   case S16_TYPE: fprintf(fp,".s16 %d\n", value.s16 ); break;
   case S32_TYPE: fprintf(fp,".s32 %d\n", value.s32 ); break;
   case S64_TYPE: fprintf(fp,".s64 %Ld\n", value.s64 ); break;
   case U8_TYPE:  fprintf(fp,".u8  %u [0x%02x]\n", value.u8, (unsigned) value.u8 );  break;
   case U16_TYPE: fprintf(fp,".u16 %u [0x%04x]\n", value.u16, (unsigned) value.u16 ); break;
   case U32_TYPE: fprintf(fp,".u32 %u [0x%08x]\n", value.u32, (unsigned) value.u32 ); break;
   case U64_TYPE: fprintf(fp,".u64 %llu [0x%llx]\n", value.u64, value.u64 ); break;
   case F16_TYPE: fprintf(fp,".f16 %f [0x%04x]\n",  value.f16, (unsigned) value.u16 ); break;
   case F32_TYPE: fprintf(fp,".f32 %.15lf [0x%08x]\n",  value.f32, value.u32 ); break;
   case F64_TYPE: fprintf(fp,".f64 %.15le [0x%016llx]\n", value.f64, value.u64 ); break;
   case B8_TYPE:  fprintf(fp,".b8  0x%02x\n",   (unsigned) value.u8 );  break;
   case B16_TYPE: fprintf(fp,".b16 0x%04x\n",   (unsigned) value.u16 ); break;
   case B32_TYPE: fprintf(fp,".b32 0x%08x\n", (unsigned) value.u32 ); break;
   case B64_TYPE: fprintf(fp,".b64 0x%llx\n",    (unsigned long long ) value.u64 ); break;
   case PRED_TYPE: fprintf(fp,".pred %u\n",     (unsigned) value.pred ); break;
   default: 
      fprintf( fp, "non-scalar type\n" );
      break;
   }
   fflush(fp);
}

static void print_reg( std::string name, ptx_reg_t value, symbol_table *symtab )
{
   print_reg(stdout,name,value,symtab);
}

void ptx_thread_info::callstack_push( unsigned pc, unsigned rpc, const symbol *return_var_src, const symbol *return_var_dst, unsigned call_uid )
{
   m_RPC = -1;
   m_RPC_updated = true;
   m_last_was_call = true;
   assert( m_func_info != NULL );
   m_callstack.push_back( stack_entry(m_symbol_table,m_func_info,pc,rpc,return_var_src,return_var_dst,call_uid) );
   m_regs.push_back( reg_map_t() );
   m_debug_trace_regs_modified.push_back( reg_map_t() );
   m_debug_trace_regs_read.push_back( reg_map_t() );
   m_local_mem_stack_pointer += m_func_info->local_mem_framesize(); 
}

//ptxplus version of callstack_push.
void ptx_thread_info::callstack_push_plus( unsigned pc, unsigned rpc, const symbol *return_var_src, const symbol *return_var_dst, unsigned call_uid )
{
   m_RPC = -1;
   m_RPC_updated = true;
   m_last_was_call = true;
   assert( m_func_info != NULL );
   m_callstack.push_back( stack_entry(m_symbol_table,m_func_info,pc,rpc,return_var_src,return_var_dst,call_uid) );
   //m_regs.push_back( reg_map_t() );
   //m_debug_trace_regs_modified.push_back( reg_map_t() );
   //m_debug_trace_regs_read.push_back( reg_map_t() );
   m_local_mem_stack_pointer += m_func_info->local_mem_framesize();
}


bool ptx_thread_info::callstack_pop()
{
   const symbol *rv_src = m_callstack.back().m_return_var_src;
   const symbol *rv_dst = m_callstack.back().m_return_var_dst;
   assert( !((rv_src != NULL) ^ (rv_dst != NULL)) ); // ensure caller and callee agree on whether there is a return value

   // read return value from callee frame
   arg_buffer_t buffer;
   if( rv_src != NULL ) 
      buffer = copy_arg_to_buffer(this, operand_info(rv_src), rv_dst );

   m_symbol_table = m_callstack.back().m_symbol_table;
   m_NPC = m_callstack.back().m_PC;
   m_RPC_updated = true;
   m_last_was_call = false;
   m_RPC = m_callstack.back().m_RPC;
   m_func_info = m_callstack.back().m_func_info;
   if( m_func_info ) {
      assert( m_local_mem_stack_pointer >= m_func_info->local_mem_framesize() );
      m_local_mem_stack_pointer -= m_func_info->local_mem_framesize(); 
   }
   m_callstack.pop_back();
   m_regs.pop_back();
   m_debug_trace_regs_modified.pop_back();
   m_debug_trace_regs_read.pop_back();

   // write return value into caller frame
   if( rv_dst != NULL ) 
      copy_buffer_to_frame(this, buffer);

   return m_callstack.empty();
}

//ptxplus version of callstack_pop
bool ptx_thread_info::callstack_pop_plus()
{
   const symbol *rv_src = m_callstack.back().m_return_var_src;
   const symbol *rv_dst = m_callstack.back().m_return_var_dst;
   assert( !((rv_src != NULL) ^ (rv_dst != NULL)) ); // ensure caller and callee agree on whether there is a return value

   // read return value from callee frame
   arg_buffer_t buffer;
   if( rv_src != NULL )
      buffer = copy_arg_to_buffer(this, operand_info(rv_src), rv_dst );

   m_symbol_table = m_callstack.back().m_symbol_table;
   m_NPC = m_callstack.back().m_PC;
   m_RPC_updated = true;
   m_last_was_call = false;
   m_RPC = m_callstack.back().m_RPC;
   m_func_info = m_callstack.back().m_func_info;
   if( m_func_info ) {
      assert( m_local_mem_stack_pointer >= m_func_info->local_mem_framesize() );
      m_local_mem_stack_pointer -= m_func_info->local_mem_framesize();
   }
   m_callstack.pop_back();
   //m_regs.pop_back();
   //m_debug_trace_regs_modified.pop_back();
   //m_debug_trace_regs_read.pop_back();

   // write return value into caller frame
   if( rv_dst != NULL )
      copy_buffer_to_frame(this, buffer);

   return m_callstack.empty();
}

void ptx_thread_info::dump_callstack() const
{
   std::list<stack_entry>::const_iterator c=m_callstack.begin();
   std::list<reg_map_t>::const_iterator r=m_regs.begin();

   printf("\n\n");
   printf("Call stack for thread uid = %u (sc=%u, hwtid=%u)\n", m_uid, m_hw_sid, m_hw_tid );
   while( c != m_callstack.end() && r != m_regs.end() ) {
      const stack_entry &c_e = *c;
      const reg_map_t &regs = *r;
      if( !c_e.m_valid ) {
         printf("  <entry>                              #regs = %zu\n", regs.size() );
      } else {
         printf("  %20s  PC=%3u RV= (callee=\'%s\',caller=\'%s\') #regs = %zu\n", 
                c_e.m_func_info->get_name().c_str(), c_e.m_PC, 
                c_e.m_return_var_src->name().c_str(), 
                c_e.m_return_var_dst->name().c_str(), 
                regs.size() );
      }
      c++;
      r++;
   }
   if( c != m_callstack.end() || r != m_regs.end() ) {
      printf("  *** mismatch in m_regs and m_callstack sizes ***\n" );
   }
   printf("\n\n");
}

std::string ptx_thread_info::get_location() const
{
   const ptx_instruction *pI = m_func_info->get_instruction(m_PC);
   char buf[1024];
   snprintf(buf,1024,"%s:%u", pI->source_file(), pI->source_line() );
   return std::string(buf);
}

const ptx_instruction *ptx_thread_info::get_inst() const
{
   return m_func_info->get_instruction(m_PC);
}

const ptx_instruction *ptx_thread_info::get_inst( addr_t pc ) const
{
   return m_func_info->get_instruction(pc);
}

int ptx_thread_info::bb_mask_cnt(int num, addr_t pc ) 
{
	bool reg_release_potential = false;
	int release = 0;
    const ptx_instruction *pI = m_func_info->get_instruction(pc);
	if(pI->m_bb_start)
	{
		int cnt = 0;
		if(pI->m_bb_mask.test(0)) cnt++;
		if(pI->m_bb_mask.test(1)) cnt++;
		if(pI->m_bb_mask.test(2)) cnt++;
		//printf("%d] FULL: bbmask_cnt = %d\n", num, cnt);
		
		return cnt;
	}else
	{
		ptx_instruction* ppI = (ptx_instruction*)pI->m_bb_start_inst;
		int cnt = 0;
		if(ppI->m_bb_mask.test(pI->m_bb_index*3+0)) cnt++;
		if(ppI->m_bb_mask.test(pI->m_bb_index*3+1)) cnt++;
		if(ppI->m_bb_mask.test(pI->m_bb_index*3+2)) cnt++;
		//printf("%d] FULL: bbmask_cnt = %d\n", num, cnt);
		return cnt;
	}
}
#ifdef RENAMING
int ptx_thread_info::check_release( addr_t pc ) 
{
	bool reg_release_potential = false;
	int release = 0;
    const ptx_instruction *pI = m_func_info->get_instruction(pc);
	if(pI->m_operands.size()>1)
		if(pI->src1().is_last_read())
			release++;
	if(pI->m_operands.size()>2)
		if(pI->src2().is_last_read())
			release++;
	if(pI->m_operands.size()>3)
		if(pI->src3().is_last_read())
			release++;
	//printf("PHYS_REG_FULL: %d | %d %d %d\n", pI->get_opcode(), release[0], release[1], release[2]);
	//return reg_release_potential;	
	return release;	
}
#endif

void ptx_thread_info::dump_regs( FILE *fp )
{
   if(m_regs.empty()) return;
   if(m_regs.back().empty()) return;
   fprintf(fp,"Register File Contents:\n");
   fflush(fp);
   reg_map_t::const_iterator r;
   for ( r=m_regs.back().begin(); r != m_regs.back().end(); ++r ) {
      const symbol *sym = r->first;
      ptx_reg_t value = r->second;
      std::string name = sym->name();
      print_reg(fp,name,value,m_symbol_table);
   }
}

void ptx_thread_info::dump_modifiedregs(FILE *fp)
{
   if( !(m_debug_trace_regs_modified.empty() || 
         m_debug_trace_regs_modified.back().empty()) ) { 
      fprintf(fp,"Output Registers:\n");
      fflush(fp);
      reg_map_t::iterator r;
      for ( r=m_debug_trace_regs_modified.back().begin(); r != m_debug_trace_regs_modified.back().end(); ++r ) {
         const symbol *sym = r->first;
         std::string name = sym->name();
         ptx_reg_t value = r->second;
         print_reg(fp,name,value,m_symbol_table);
      }
   }
   if( !(m_debug_trace_regs_read.empty() ||
         m_debug_trace_regs_read.back().empty()) ) { 
      fprintf(fp,"Input Registers:\n");
      fflush(fp);
      reg_map_t::iterator r;
      for ( r=m_debug_trace_regs_read.back().begin(); r != m_debug_trace_regs_read.back().end(); ++r ) {
         const symbol *sym = r->first;
         std::string name = sym->name();
         ptx_reg_t value = r->second;
         print_reg(fp,name,value,m_symbol_table);
      }
   }
}

void ptx_thread_info::push_breakaddr(const operand_info &breakaddr) 
{
   m_breakaddrs.push(breakaddr);
}

const operand_info& ptx_thread_info::pop_breakaddr() 
{
   if(m_breakaddrs.empty()) {
      printf("empty breakaddrs stack");
      assert(0);
   }
   operand_info& breakaddr = m_breakaddrs.top();
   m_breakaddrs.pop();
   return breakaddr;
}

void ptx_thread_info::set_npc( const function_info *f )
{
   m_NPC = f->get_start_PC();
   m_func_info = const_cast<function_info*>( f );
   m_symbol_table = m_func_info->get_symtab();
}


void feature_not_implemented( const char *f ) 
{
   printf("GPGPU-Sim: feature '%s' not supported\n", f );
   abort();
}
