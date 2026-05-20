GUIDELINE: Running Two vLLM Servers on Two Separate GPU Nodes Using a Multi‑Node SLURM Allocation

Goal:
Allocate 2 GPU nodes (each with 4× V100) inside the same SLURM job. Start one vLLM server on each GPU node. Both servers become reachable from any other node inside the same SLURM allocation. Codex or other clients can call either server by hostname and port.

1. Request a 2‑node GPU allocation:
   salloc -N 2 -p multigpu --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=24:00:00 --exclusive

2. Identify the two GPU nodes assigned:
   scontrol show hostnames $SLURM_JOB_NODELIST
   Let:
     GPU_NODE_A = first node
     GPU_NODE_B = second node

3. Start vLLM server #1 on GPU_NODE_A:
   srun -N1 -n1 -w $GPU_NODE_A bash -lc "
     export VLLM_CPU_BIND=none
     vllm serve Qwen2.5-Coder-32B-Instruct \
       --host 0.0.0.0 \
       --port 8011 \
       --tensor-parallel-size 4 \
       --max-model-len 128000 \
       > vllm_A.out 2> vllm_A.err &
   "

4. Start vLLM server #2 on GPU_NODE_B:
   srun -N1 -n1 -w $GPU_NODE_B bash -lc "
     export VLLM_CPU_BIND=none
     vllm serve Qwen2.5-Coder-32B-Instruct \
       --host 0.0.0.0 \
       --port 8012 \
       --tensor-parallel-size 4 \
       --max-model-len 128000 \
       > vllm_B.out 2> vllm_B.err &
   "

5. Both servers are now reachable from any node inside the same SLURM job:
   Server A endpoint: http://$GPU_NODE_A:8011/v1
   Server B endpoint: http://$GPU_NODE_B:8012/v1

6. To run Codex or any client inside this same job:
   srun -N1 -n1 -w <ANY_NODE_IN_JOB> bash -lc "
     export SERVER_A=http://$GPU_NODE_A:8011/v1
     export SERVER_B=http://$GPU_NODE_B:8012/v1
     python codex_orchestrator.py
   "

7. Important: Cross-node networking only works because all nodes are inside the same SLURM allocation. Explorer blocks node-to-node TCP unless nodes share the same job. Do not attempt to access these ports from outside the job.

8. CPU binding must be disabled for vLLM on Explorer:
   --cpu-bind=none
   export VLLM_CPU_BIND=none

9. Summary:
   - Allocate 2 GPU nodes in one SLURM job.
   - Launch vLLM on each node with different ports.
   - Use hostnames assigned by SLURM to reach each server.
   - Codex or other clients can call either server inside the same job.
   - This setup allows two independent Qwen2.5-Coder-32B inference servers running simultaneously on two V100 nodes.
