// Copyright (c) 2026
// DAE (Decoupled Access-Execute) Access Processor for GPGPU-Sim
// Adds a dedicated per-SM Access Processor that runs ahead of the
// Execute Processor to identify global loads and push FIFO tokens.
//
// The AP scans ahead of the EP in program order, identifying global loads.
// For each load found, it immediately pushes a token to a per-warp FIFO.
// The EP checks the FIFO before issuing loads — if a token is present,
// it releases the scoreboard early (allowing dependent instructions to
// proceed) but still issues the load to L1/DRAM normally. This enables
// multiple loads to become concurrent in the memory system.

#ifndef DAE_AP_H
#define DAE_AP_H

#include <vector>
#include "../abstract_hardware_model.h"

class shader_core_ctx;
class shader_core_config;

// Per-warp FIFO between AP and EP (circular buffer of tokens)
class dae_fifo {
 public:
  dae_fifo() : m_head(0), m_tail(0), m_count(0), m_depth(0) {}
  dae_fifo(unsigned depth)
      : m_head(0), m_tail(0), m_count(0), m_depth(depth) {}

  void init(unsigned depth) {
    m_depth = depth;
    m_head = 0;
    m_tail = 0;
    m_count = 0;
  }

  bool push(unsigned long long cycle) {
    if (is_full()) return false;
    m_tail = (m_tail + 1) % (m_depth + 1);
    m_count++;
    return true;
  }

  bool pop() {
    if (is_empty()) return false;
    m_head = (m_head + 1) % (m_depth + 1);
    m_count--;
    return true;
  }

  bool is_full() const { return m_count >= m_depth; }
  bool is_empty() const { return m_count == 0; }
  unsigned occupancy() const { return m_count; }
  unsigned depth() const { return m_depth; }

  void flush() {
    m_head = 0;
    m_tail = 0;
    m_count = 0;
  }

 private:
  unsigned m_head;
  unsigned m_tail;
  unsigned m_count;
  unsigned m_depth;
};

// Per-warp AP state
struct dae_ap_warp_state {
  address_type ap_pc;
  bool active;
  unsigned loads_issued;

  dae_ap_warp_state() : ap_pc(0), active(false), loads_issued(0) {}

  void reset() {
    ap_pc = 0;
    active = false;
    loads_issued = 0;
  }
};

// Access Processor unit per SM
class dae_ap_unit {
 public:
  dae_ap_unit(shader_core_ctx *core, const shader_core_config *config,
              unsigned num_warps, unsigned fifo_depth);

  void cycle();

  void activate_warp(unsigned wid, address_type start_pc);
  void deactivate_warp(unsigned wid);
  void flush_warp(unsigned wid);

  // FIFO interface for EP
  bool fifo_has_data(unsigned wid) const;
  bool fifo_pop(unsigned wid);
  unsigned fifo_depth() const { return m_fifo_depth; }

  void print_stats(FILE *fout) const;

  // Stats getters for aggregation
  unsigned long long get_loads_issued() const { return m_loads_issued; }
  unsigned long long get_stall_cycles() const { return m_stall_cycles; }
  unsigned long long get_fifo_pushes() const { return m_fifo_pushes; }
  unsigned long long get_fifo_pops() const { return m_fifo_pops; }
  unsigned long long get_fifo_full_stalls() const { return m_fifo_full_stalls; }

 private:
  void advance_warps();

  shader_core_ctx *m_core;
  const shader_core_config *m_config;
  unsigned m_num_warps;
  unsigned m_fifo_depth;

  std::vector<dae_ap_warp_state> m_warp_state;
  std::vector<dae_fifo> m_fifos;

  // Max instructions AP can be ahead of EP
  static const unsigned MAX_RUNAHEAD = 256;

  // Stats
  unsigned long long m_loads_issued;
  unsigned long long m_stall_cycles;
  unsigned long long m_fifo_pushes;
  unsigned long long m_fifo_pops;
  unsigned long long m_fifo_full_stalls;
};

#endif  // DAE_AP_H
