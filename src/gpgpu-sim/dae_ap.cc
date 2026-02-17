// Copyright (c) 2026
// DAE (Decoupled Access-Execute) Access Processor for GPGPU-Sim

#include "dae_ap.h"
#include <cstdio>
#include "gpu-sim.h"
#include "shader.h"

dae_ap_unit::dae_ap_unit(shader_core_ctx *core,
                         const shader_core_config *config, unsigned num_warps,
                         unsigned fifo_depth)
    : m_core(core),
      m_config(config),
      m_num_warps(num_warps),
      m_fifo_depth(fifo_depth),
      m_loads_issued(0),
      m_stall_cycles(0),
      m_fifo_pushes(0),
      m_fifo_pops(0),
      m_fifo_full_stalls(0) {
  m_warp_state.resize(num_warps);
  m_fifos.resize(num_warps);
  for (unsigned i = 0; i < num_warps; i++) {
    m_fifos[i].init(fifo_depth);
  }
}

void dae_ap_unit::activate_warp(unsigned wid, address_type start_pc) {
  m_warp_state[wid].reset();
  m_warp_state[wid].active = true;
  m_warp_state[wid].ap_pc = start_pc;
  m_fifos[wid].flush();
}

void dae_ap_unit::deactivate_warp(unsigned wid) {
  m_warp_state[wid].active = false;
  m_fifos[wid].flush();
}

void dae_ap_unit::flush_warp(unsigned wid) {
  m_warp_state[wid].reset();
  m_fifos[wid].flush();
}

bool dae_ap_unit::fifo_has_data(unsigned wid) const {
  return !m_fifos[wid].is_empty();
}

bool dae_ap_unit::fifo_pop(unsigned wid) {
  if (m_fifos[wid].pop()) {
    m_fifo_pops++;
    return true;
  }
  return false;
}

void dae_ap_unit::cycle() {
  advance_warps();
}

void dae_ap_unit::advance_warps() {
  unsigned long long cur_cycle = m_core->get_cycle();

  for (unsigned w = 0; w < m_num_warps; w++) {
    dae_ap_warp_state &ws = m_warp_state[w];
    if (!ws.active) continue;

    // Check if warp is done
    if (m_core->is_warp_done(w)) {
      ws.active = false;
      continue;
    }

    // Check if FIFO is full
    if (m_fifos[w].is_full()) {
      m_fifo_full_stalls++;
      continue;
    }

    // Check AP is not too far ahead of EP
    address_type ep_pc = m_core->get_warp_pc(w);
    if (ws.ap_pc > ep_pc && (ws.ap_pc - ep_pc) > MAX_RUNAHEAD) {
      continue;
    }

    // Fetch instruction at AP PC
    const warp_inst_t *pI = m_core->dae_get_next_inst(w, ws.ap_pc);
    if (pI == NULL) {
      ws.active = false;
      continue;
    }

    // Check if this is a global load
    if ((pI->op == LOAD_OP || pI->op == TENSOR_CORE_LOAD_OP) &&
        pI->space.is_global()) {
      // Push FIFO token immediately (no latency modeling)
      m_fifos[w].push(cur_cycle);
      m_fifo_pushes++;

      ws.loads_issued++;
      m_loads_issued++;

      // Advance AP PC past this instruction
      ws.ap_pc += pI->isize;
    } else if (pI->op == STORE_OP || pI->op == TENSOR_CORE_STORE_OP) {
      // Stores: AP just advances past them
      ws.ap_pc += pI->isize;
    } else if (pI->op == EXIT_OPS) {
      ws.active = false;
    } else if (pI->op == BARRIER_OP) {
      ws.active = false;
    } else if (pI->op == BRANCH_OP) {
      if (pI->branch_target_pc != 0 && pI->branch_target_pc < ws.ap_pc) {
        // Loop back-edge: follow the loop
        ws.ap_pc = pI->branch_target_pc;
      } else if (pI->branch_target_pc != 0 &&
                 pI->branch_target_pc >= ws.ap_pc) {
        // Forward branch: take fall-through path
        ws.ap_pc += pI->isize;
      } else {
        // Unknown branch target: stop conservatively
        ws.active = false;
      }
    } else if (pI->op == RET_OPS || pI->op == CALL_OPS) {
      ws.active = false;
    } else {
      // ALU, SFU, etc: AP just advances past them
      ws.ap_pc += pI->isize;
    }
  }
}

void dae_ap_unit::print_stats(FILE *fout) const {
  fprintf(fout, "dae_ap_loads_issued = %llu\n", m_loads_issued);
  fprintf(fout, "dae_ap_stall_cycles = %llu\n", m_stall_cycles);
  fprintf(fout, "dae_ap_fifo_pushes = %llu\n", m_fifo_pushes);
  fprintf(fout, "dae_ap_fifo_pops = %llu\n", m_fifo_pops);
  fprintf(fout, "dae_ap_fifo_full_stalls = %llu\n", m_fifo_full_stalls);
}
