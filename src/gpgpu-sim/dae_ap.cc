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
      m_fifo_full_stalls(0),
      m_dep_stalls(0) {
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
  // Remove pending loads for this warp
  auto it = m_pending_loads.begin();
  while (it != m_pending_loads.end()) {
    if (it->warp_id == wid)
      it = m_pending_loads.erase(it);
    else
      ++it;
  }
}

void dae_ap_unit::flush_warp(unsigned wid) {
  m_warp_state[wid].reset();
  m_fifos[wid].flush();
  auto it = m_pending_loads.begin();
  while (it != m_pending_loads.end()) {
    if (it->warp_id == wid)
      it = m_pending_loads.erase(it);
    else
      ++it;
  }
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
  unsigned long long cur_cycle = m_core->get_cycle();

  // Phase 1: Complete pending loads that have reached their ready cycle
  process_pending_loads();

  // Phase 2: For each active warp, try to advance the AP
  advance_warps();
}

void dae_ap_unit::process_pending_loads() {
  unsigned long long cur_cycle = m_core->get_cycle();

  auto it = m_pending_loads.begin();
  while (it != m_pending_loads.end()) {
    if (it->ready_cycle <= cur_cycle) {
      unsigned wid = it->warp_id;
      unsigned dest_reg = it->dest_reg;

      // Try to push token to FIFO
      if (!m_fifos[wid].is_full()) {
        m_fifos[wid].push(cur_cycle);
        m_fifo_pushes++;

        // Clear pending register
        m_warp_state[wid].pending_regs.erase(dest_reg);

        // Unstall if this was the blocking register
        if (m_warp_state[wid].stalled &&
            m_warp_state[wid].stall_reg == (int)dest_reg) {
          m_warp_state[wid].stalled = false;
          m_warp_state[wid].stall_reg = -1;
        }

        it = m_pending_loads.erase(it);
      } else {
        // FIFO full, can't complete this load yet
        m_fifo_full_stalls++;
        ++it;
      }
    } else {
      ++it;
    }
  }
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

    // Check if AP is stalled on a dependency
    if (ws.stalled) {
      // Check if stall resolved
      if (ws.pending_regs.find(ws.stall_reg) == ws.pending_regs.end()) {
        ws.stalled = false;
        ws.stall_reg = -1;
      } else {
        m_stall_cycles++;
        continue;
      }
    }

    // Check AP is not too far ahead of EP
    address_type ep_pc = m_core->get_warp_pc(w);
    if (ws.ap_pc > ep_pc && (ws.ap_pc - ep_pc) > MAX_RUNAHEAD) {
      continue;
    }

    // Fetch instruction at AP PC
    const warp_inst_t *pI = m_core->dae_get_next_inst(w, ws.ap_pc);
    if (pI == NULL) {
      // No more instructions at this PC (end of program or invalid PC)
      ws.active = false;
      continue;
    }

    // Check if this is a global load
    if ((pI->op == LOAD_OP || pI->op == TENSOR_CORE_LOAD_OP) &&
        pI->space.is_global()) {
      // Check source register dependencies against pending AP loads
      bool has_dep = false;
      for (unsigned i = 0; i < MAX_INPUT_VALUES && i < 24; i++) {
        if (pI->in[i] > 0 &&
            ws.pending_regs.find(pI->in[i]) != ws.pending_regs.end()) {
          // Source register depends on a pending AP load
          ws.stalled = true;
          ws.stall_reg = pI->in[i];
          has_dep = true;
          m_dep_stalls++;
          break;
        }
      }
      if (has_dep) continue;

      // Issue this load: model a fixed memory latency
      unsigned dest_reg = 0;
      for (unsigned r = 0; r < 8; r++) {
        if (pI->out[r] > 0) {
          dest_reg = pI->out[r];
          ws.pending_regs.insert(dest_reg);
        }
      }

      // Create pending load entry
      dae_pending_load pl;
      pl.warp_id = w;
      pl.dest_reg = dest_reg;
      pl.ready_cycle = cur_cycle + AP_LOAD_LATENCY;
      m_pending_loads.push_back(pl);

      ws.loads_issued++;
      m_loads_issued++;

      // Advance AP PC past this instruction
      ws.ap_pc += pI->isize;
    } else if (pI->op == STORE_OP || pI->op == TENSOR_CORE_STORE_OP) {
      // Stores: AP just advances past them
      // But check source deps first (store address may depend on pending load)
      bool has_dep = false;
      for (unsigned i = 0; i < MAX_INPUT_VALUES && i < 24; i++) {
        if (pI->in[i] > 0 &&
            ws.pending_regs.find(pI->in[i]) != ws.pending_regs.end()) {
          ws.stalled = true;
          ws.stall_reg = pI->in[i];
          has_dep = true;
          m_dep_stalls++;
          break;
        }
      }
      if (has_dep) continue;
      ws.ap_pc += pI->isize;
    } else if (pI->op == EXIT_OPS) {
      // End of program
      ws.active = false;
    } else if (pI->op == BARRIER_OP) {
      // AP stops at barriers
      ws.active = false;
    } else if (pI->op == BRANCH_OP) {
      // Branch handling for AP:
      // Use branch_target_pc (resolved during PTX assembly) to detect
      // loop back-edges vs forward branches.
      if (pI->branch_target_pc != 0 && pI->branch_target_pc < ws.ap_pc) {
        // Loop back-edge: target is before current PC.
        // AP follows the loop by jumping to the target (loop head).
        // This allows the AP to prefetch loads for all loop iterations.
        ws.ap_pc = pI->branch_target_pc;
      } else if (pI->branch_target_pc != 0 &&
                 pI->branch_target_pc >= ws.ap_pc) {
        // Forward branch: AP takes the fall-through path.
        // This is an approximation - AP assumes the branch is not taken,
        // which is correct for predicated forward branches that skip
        // optional code (e.g., early exit checks, remainder handling).
        ws.ap_pc += pI->isize;
      } else {
        // Unknown branch target: stop conservatively
        ws.active = false;
      }
    } else if (pI->op == RET_OPS || pI->op == CALL_OPS) {
      // RET/CALL: AP stops
      ws.active = false;
    } else {
      // ALU, SFU, etc: AP just advances past them (1 cycle model)
      // Check if any output registers need to be cleared from pending
      // (they're being overwritten by a non-load instruction)
      for (unsigned r = 0; r < 8; r++) {
        if (pI->out[r] > 0) {
          ws.pending_regs.erase(pI->out[r]);
        }
      }
      ws.ap_pc += pI->isize;
    }
  }
}

void dae_ap_unit::print_stats(FILE *fout) const {
  fprintf(fout, "dae_ap_loads_issued = %llu\n", m_loads_issued);
  fprintf(fout, "dae_ap_stall_cycles = %llu\n", m_stall_cycles);
  fprintf(fout, "dae_ap_dep_stalls = %llu\n", m_dep_stalls);
  fprintf(fout, "dae_ap_fifo_pushes = %llu\n", m_fifo_pushes);
  fprintf(fout, "dae_ap_fifo_pops = %llu\n", m_fifo_pops);
  fprintf(fout, "dae_ap_fifo_full_stalls = %llu\n", m_fifo_full_stalls);
}
