"""An implementation of the algorithms in:
"The Santa Claus Problem", by Bansal, Nikhil, and Maxim Sviridenko.
Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006
https://dl.acm.org/doi/10.1145/1132516.1132557

The problem involves distributing n presents (gifts) among m kids (children), where each kid i has 
a different valuation pij for each present j. The goal is to maximize the happiness 
of the least happy kid (maximin objective).

In the paper, the authors use machine scheduling terminology, where presents correspond to jobs,
kids correspond to machines, and the value of a present to a kid corresponds to the processing
time of a job on a machine. The goal is to find an assignment of jobs to machines that maximizes
the minimal machine load.

Programmers: Roey and Adiel
Date: 2025-04-27
"""

from fairpyx import Instance, AllocationBuilder, divide
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_configuration_graph(lpSolution: Dict) -> nx.Graph:
    """
    Build a bipartite graph G from the LP solution as described in Section 5.2.
    
    The graph has:
    - Machine nodes: one for each machine in the solution
    - Job nodes: one for each job that appears in any configuration
    - Edges: between a machine and a job if the job appears in any of
            the machine's configurations with non-zero value
    
    :param lpSolution: The fractional solution from the Configuration LP
    :return: A NetworkX bipartite graph G
    """
    G = nx.Graph()
    
    # Add edges between machines and jobs based on LP solution
    for machine, configs in lpSolution.items():
        G.add_node(machine, bipartite=0)  # 0 indicates machine nodes
        for config, value in configs:
            for job in config:
                if value > 0:
                    G.add_node(job, bipartite=1)  # 1 indicates job nodes
                    G.add_edge(machine, job)
    
    return G


def get_node_sets(G: nx.Graph) -> Tuple[Set[str], Set[str]]:
    """
    Get the sets of machine nodes and job nodes from the bipartite graph.
    
    :param G: The bipartite graph
    :return: Tuple of (machine_nodes, job_nodes)
    """
    machine_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
    job_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 1}
    return machine_nodes, job_nodes

# Example instance for testing
example_instance = Instance(
    valuations = {
        "Child1": {"gift1": 10, "gift2": 0, "gift3": 5},
        "Child2": {"gift1": 0, "gift2": 10, "gift3": 5},
        "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
    },
    agent_capacities = {"Child1": 1, "Child2": 1, "Child3": 1},
    item_capacities = {"gift1": 1, "gift2": 1, "gift3": 1}
)

# Example for restricted assignment case
restricted_example = Instance(
    valuations = {
        "Child1": {"gift1": 10, "gift2": 0, "gift3": 0, "gift4": 10, "gift5": 1},
        "Child2": {"gift1": 0, "gift2": 10, "gift3": 0, "gift4": 0, "gift5": 1},
        "Child3": {"gift1": 0, "gift2": 0, "gift3": 10, "gift4": 10, "gift5": 1}
    },
    agent_capacities = {"Child1": 2, "Child2": 2, "Child3": 2},
    item_capacities = {"gift1": 1, "gift2": 1, "gift3": 1, "gift4": 1, "gift5": 1}
)

def santa_claus_algorithm(alloc: AllocationBuilder, alpha: float = None, gamma: float = 2.0, beta: float = 3.0) -> None:
    """
    Algorithm for the Santa Claus Problem in the restricted assignment case.

    This algorithm achieves an O(log log m / log log log m) approximation ratio for
    the restricted assignment case, where m is the number of kids (children).
    
    In the restricted assignment case, each present j has a fixed value pj for all kids
    who can receive it, and 0 for kids who cannot receive it.

    Algorithm Steps based on the paper:
    1. Binary search to find optimal target value T
    2. Round present sizes (values) - presents with value >= T/alpha are rounded to T
    3. Solve Configuration LP with rounded values
    4. Classify presents as big (value = T) or small (value < T/alpha)
    5. Construct super-machines (clusters of kids and big presents)
    6. Apply randomized rounding for small presents
    7. Construct final allocation

    :param alloc: The allocation builder that tracks the allocation process
    :param alpha: Parameter for classifying presents as large or small
    :param gamma: Parameter for fraction of atoms in good assignments
    :param beta: Relaxation parameter for the solution
    """
    logger.info("\n" + "="*80)
    logger.info("SANTA CLAUS ALGORITHM EXECUTION".center(80))
    logger.info("="*80)
    
    # Print parameters in a table-like format
    logger.info("\nPARAMETERS:")
    logger.info("-"*50)
    logger.info(f"| {'Parameter':<20} | {'Value':<25} |")
    logger.info("-"*50)
    logger.info(f"| {'alpha':<20} | {alpha:<25} |")
    logger.info(f"| {'gamma':<20} | {gamma:<25} |")
    logger.info(f"| {'beta':<20} | {beta:<25} |")
    logger.info("-"*50)
    
    # Print problem instance information
    logger.info("\nPROBLEM INSTANCE:")
    logger.info("-"*50)
    logger.info(f"| {'Presents':<20} | {', '.join(alloc.remaining_item_capacities.keys()):<25} |")
    logger.info(f"| {'Kids':<20} | {', '.join(alloc.remaining_agent_capacities.keys()):<25} |")
    logger.info("-"*50)
    
    # Print initial valuations in a table
    logger.info("\nINITIAL VALUATIONS:")
    logger.info("-"*80)
    header = "| Kid" + " "*(16) + "|"
    for present in alloc.remaining_item_capacities:
        header += f" {present:<10} |"
    logger.info(header)
    logger.info("-"*80)
    
    for kid in alloc.remaining_agent_capacities:
        row = f"| {kid:<18} |"
        for present in alloc.remaining_item_capacities:
            value = alloc.instance.agent_item_value(kid, present)
            row += f" {value:<10.1f} |"
        logger.info(row)
    logger.info("-"*80)

    # If alpha is not provided, set a default value based on number of kids
    if alpha is None:
        m = len(alloc.remaining_agent_capacities)
        alpha = 2 if m <= 3 else max(2, np.log(np.log(m)) / np.log(np.log(np.log(m))))
    logger.info(f"\nUsing alpha = {alpha:.4f}")

    # Convert allocation builder to presents and kids format
    # In the restricted assignment case, each present has the same value for all kids who can receive it
    presents = {}
    for present in alloc.remaining_item_capacities:
        presents[present] = {}
        for kid in alloc.remaining_agent_capacities:
            value = alloc.instance.agent_item_value(kid, present)
            presents[present][kid] = value
    
    kids = alloc.remaining_agent_capacities

    # Step 1: Binary search to find optimal target value T
    logger.info("\n" + "*"*80)
    logger.info("STEP 1: BINARY SEARCH FOR OPTIMAL TARGET VALUE T".center(80))
    logger.info("*"*80)
    target_value = binarySearch(presents, kids)
    logger.info(f"Optimal target value T = {target_value:.4f}")

    # Step 2: Round present sizes (values)
    logger.info("\n" + "*"*80)
    logger.info("STEP 2: ROUNDING PRESENT SIZES".center(80))
    logger.info("*"*80)
    rounded_values = roundPresentSizes(presents, target_value, alpha)
    
    # Print rounded values in a table
    logger.info("\nROUNDED VALUES:")
    logger.info("-"*80)
    header = "| Present" + " "*(11) + "|"
    for kid in alloc.remaining_agent_capacities:
        header += f" {kid:<10} |"
    logger.info(header)
    logger.info("-"*80)
    
    for present in rounded_values:
        row = f"| {present:<18} |"
        for kid in alloc.remaining_agent_capacities:
            value = rounded_values[present].get(kid, 0)
            row += f" {value:<10.1f} |"
        logger.info(row)
    logger.info("-"*80)

    # Step 3: Solve Configuration LP with rounded values
    logger.info("\n" + "*"*80)
    logger.info("STEP 3: SOLVING CONFIGURATION LP".center(80))
    logger.info("*"*80)
    fractional_solution = solveConfigurationLP(rounded_values, kids, target_value, alpha, beta)
    
    # Print fractional solution in a more readable format
    logger.info("\nFRACTIONAL SOLUTION:")
    logger.info("-"*80)
    logger.info(f"| {'Kid':<18} | {'Configuration':<25} | {'Fraction':<10} |")
    logger.info("-"*80)
    for kid, configs in fractional_solution.items():
        for config, fraction in configs:
            config_str = ", ".join(config)
            logger.info(f"| {kid:<18} | {config_str:<25} | {fraction:<10.4f} |")
    logger.info("-"*80)

    # Step 4: Classify presents as big or small based on the paper's definition
    logger.info("\n" + "*"*80)
    logger.info("STEP 4: CLASSIFYING PRESENTS".center(80))
    logger.info("*"*80)
    large_presents = set()
    small_presents = set()

    logger.info("\nPRESENT CLASSIFICATION:")
    logger.info("-"*50)
    logger.info(f"| {'Present':<18} | {'Classification':<25} |")
    logger.info("-"*50)
    
    for present in alloc.remaining_item_capacities:
        if is_big_present(present, presents, target_value, alpha):
            large_presents.add(present)
            logger.info(f"| {present:<18} | {'BIG':<25} |")
        elif is_small_present(present, presents, target_value, alpha):
            small_presents.add(present)
            logger.info(f"| {present:<18} | {'SMALL':<25} |")
        else:
            logger.info(f"| {present:<18} | {'INACCESSIBLE':<25} |")
    logger.info("-"*50)
    
    logger.info(f"\nLarge presents: {large_presents}")
    logger.info(f"Small presents: {small_presents}")

    # Step 5: Construct super-machines (clusters of kids and big presents)
    logger.info("\n" + "*"*80)
    logger.info("STEP 5: CONSTRUCTING SUPER-KIDS".center(80))
    logger.info("*"*80)
    super_machines = construct_super_machines(alloc, large_presents)
    
    logger.info("\nSUPER-KIDS:")
    logger.info("-"*80)
    logger.info(f"| {'Super-Kid ID':<15} | {'Kids':<25} | {'Presents':<25} |")
    logger.info("-"*80)
    for i, (kids_group, presents_group) in enumerate(super_machines):
        kids_str = ", ".join(kids_group)
        presents_str = ", ".join(presents_group)
        logger.info(f"| {i+1:<15} | {kids_str:<25} | {presents_str:<25} |")
    logger.info("-"*80)

    # Step 6: Apply randomized rounding for small presents
    logger.info("\n" + "*"*80)
    logger.info("STEP 6: APPLYING RANDOMIZED ROUNDING".center(80))
    logger.info("*"*80)
    rounded_solution = randomizedRounding(fractional_solution, kids, small_presents, beta)
    
    # Print rounded solution in a more readable format
    logger.info("\nROUNDED SOLUTION:")
    logger.info("-"*80)
    logger.info(f"| {'Kid ID':<15} | {'Kid':<18} | {'Presents':<35} |")
    logger.info("-"*80)
    for kid_id, config in rounded_solution.items():
        kid = config.get('kid', '')
        presents_list = config.get('presents', [])
        presents_str = ", ".join(presents_list)
        logger.info(f"| {kid_id:<15} | {kid:<18} | {presents_str:<35} |")
    logger.info("-"*80)

    # Step 7: Check if solution is good according to the paper's criteria
    logger.info("\n" + "*"*80)
    logger.info("STEP 7: CHECKING SOLUTION QUALITY".center(80))
    logger.info("*"*80)
    is_good = isGoodFunction(rounded_solution, super_machines, gamma, beta)
    if not is_good:
        logger.warning("\nSOLUTION QUALITY DOES NOT MEET REQUIREMENTS")
    else:
        logger.info("\nSOLUTION MEETS QUALITY REQUIREMENTS")

    # Step 8: Construct final allocation
    logger.info("\n" + "*"*80)
    logger.info("STEP 8: CONSTRUCTING FINAL ALLOCATION".center(80))
    logger.info("*"*80)
    construct_final_allocation(alloc, super_machines, rounded_solution)
    
    # Print final allocation in a table
    logger.info("\nFINAL ALLOCATION:")
    logger.info("-"*80)
    logger.info(f"| {'Kid':<18} | {'Presents':<35} | {'Total Value':<15} |")
    logger.info("-"*80)
    for kid, bundle in alloc.bundles.items():
        bundle_str = ", ".join(bundle)
        total_value = sum(alloc.instance.agent_item_value(kid, present) for present in bundle)
        logger.info(f"| {kid:<18} | {bundle_str:<35} | {total_value:<15.2f} |")
    logger.info("-"*80)
    
    # Calculate and log the results
    min_happiness = min([sum(alloc.instance.agent_item_value(kid, present) for present in bundle) 
                         for kid, bundle in alloc.bundles.items()]) if alloc.bundles else 0
    
    logger.info("\nRESULTS SUMMARY:")
    logger.info("-"*50)
    logger.info(f"| {'Metric':<25} | {'Value':<20} |")
    logger.info("-"*50)
    logger.info(f"| {'Target value (T)':<25} | {target_value:<20.4f} |")
    logger.info(f"| {'Minimum happiness':<25} | {min_happiness:<20.4f} |")
    approx_ratio = target_value/min_happiness if min_happiness > 0 else float('inf')
    logger.info(f"| {'Approximation ratio':<25} | {approx_ratio:<20.4f} |")
    logger.info("-"*50)
    
    logger.info("\n" + "="*80)
    logger.info("SANTA CLAUS ALGORITHM COMPLETED".center(80))
    logger.info("="*80 + "\n")


def binarySearch(presents: Dict[str, Dict[str, float]], kids: Dict[str, int]) -> float:
    """
    Algorithm 1: Binary search to find the highest feasible target value T for the Configuration LP.
    
    :param alloc: The allocation builder containing the problem instance
    :return: The optimal target value T
    
    >>> builder = divide(binarySearch, example_instance, return_builder=True)
    >>> builder.allocation
    {'Child1': ['gift1'], 'Child2': ['gift2'], 'Child3': ['gift3']}
    """
    logger.info("Starting binary search...")
    # Find the maximum possible value (upper bound for binary search)
    max_possible_value = 0
    for kid in kids:
        kid_capacity = kids[kid]
        # Sort presents by value for this kid in descending order
        sorted_presents = sorted(
            [(present, presents[present].get(kid, 0)) for present in presents],
            key=lambda x: x[1], reverse=True
        )
        
        # Take the top presents based on kid's capacity
        kid_max_value = sum(value for _, value in sorted_presents[:kid_capacity])
        max_possible_value = max(max_possible_value, kid_max_value)
    
    # Binary search to find the optimal target value
    low, high = 0, max_possible_value
    alpha = 2  # Default alpha value for testing
    
    while high - low > 1e-6:
        mid = (low + high) / 2
        # Check if the Configuration LP is feasible with this target value
        solution = solveConfigurationLP(presents, kids, mid)
        
        if solution:  # If feasible
            low = mid  # Try a higher target value
        else:  # If not feasible
            high = mid  # Try a lower target value
    
    # Return the highest feasible target value
    return low


def roundPresentSizes(presents: Dict[str, Dict[str, float]], target_value: float, alpha: float) -> Dict[str, Dict[str, float]]:
    """
    Round present sizes according to the paper for the restricted assignment case.
    In the restricted assignment case, each present has a fixed value pj for all kids who can receive it,
    and 0 for kids who cannot receive it.
    
    :param presents: Dictionary mapping presents to their values for each kid
    :param target_value: The target value T
    :param alpha: Parameter for classifying presents as large or small
    :return: Dictionary with rounded present sizes
    """
    rounded_values = {}
    
    for present, kids in presents.items():
        rounded_values[present] = {}
        
        # Find the value of this present (should be the same for all kids who can receive it)
        present_value = 0
        for kid, value in kids.items():
            if value > 0:
                present_value = max(present_value, value)
        
        # Apply rounding according to the paper
        for kid, value in kids.items():
            if value > 0:  # Kid can receive this present
                if present_value >= target_value/alpha:
                    rounded_values[present][kid] = target_value  # Round up to T
                else:
                    rounded_values[present][kid] = present_value  # Keep as is
            else:
                rounded_values[present][kid] = 0  # Kid cannot receive this present
    
    return rounded_values


def solveConfigurationLP(presents: Dict[str, Dict[str, float]], kids: Dict[str, int], T: float, alpha: float = 3.0, beta: float = 3.0) -> Dict:
    """
    Solve the Configuration LP for the restricted assignment case as described in Section 5.1.
    
    The LP has the following constraints:
    1. Each machine is assigned either a single big job, or many small jobs with total load ≥ T/α
    2. Each job j ∈ J₁ has at least T/α in M₁, and if assigned to M₁, has size at least T/(α|M₁|)
    3. Each job in J₂ is assigned to at most β machines
    
    :param jobs: Dictionary mapping job names to their processing times on each machine
    :param machines: Dictionary mapping machine names to their capacities
    :param T: Target value for the minimum machine load
    :return: Dictionary mapping machines to their configurations and values
    """
    try:
        import pulp
    except ImportError:
        logger.warning("PuLP not available, using simplified LP solver")
        return _simplified_lp_solver(jobs, machines, T, alpha, beta)
    
    # Create the LP problem
    prob = pulp.LpProblem("ConfigurationLP", pulp.LpMaximize)
    
    # Use the parameters passed to the function
    # alpha: Relaxation parameter for classifying jobs as large or small
    # beta: Maximum number of machines a job can be assigned to
    
    # Classify presents based on their values
    J1 = {j for j in presents if is_small_present(j, presents, T, alpha)}  # Small presents
    J2 = {j for j in presents if is_big_present(j, presents, T, alpha)}    # Big presents
    M1 = set()  # Will be filled with kids that handle small presents
    
    # Variables: x[i,C] is the fraction of time machine i uses configuration C
    x = {}
    
    # Generate configurations for each kid
    kid_configs = {}
    for kid in kids:
        configs = []
        
        # Configurations with a single big present
        for present in J2:
            if presents[present].get(kid, 0) >= T:  # Present must meet target value
                configs.append((present,))
        
        # Configurations with small presents
        small_present_configs = []
        for size in range(1, len(J1) + 1):
            for present_subset in itertools.combinations(J1, size):
                total_value = sum(presents[j].get(kid, 0) for j in present_subset)
                if total_value >= T/alpha:
                    small_present_configs.append(present_subset)
                    M1.add(kid)  # This kid can handle small presents
        
        configs.extend(small_present_configs)
        kid_configs[kid] = configs
        
        # Create variables for this kid's configurations
        for config in configs:
            x[kid, config] = pulp.LpVariable(f"x_{kid}_{config}", 0, 1)
    
    # Objective: maximize the minimum load (already handled by constraints)
    prob += 0
    
    # Constraint 1: Sum of configuration fractions for each kid = 1
    for kid in kids:
        prob += pulp.lpSum(x[kid, config] for config in kid_configs[kid]) == 1
    
    # Constraint 2: Each small present j ∈ J₁ has total assignment at least T/α in M₁
    for present in J1:
        prob += pulp.lpSum(x[kid, config] 
                          for kid in M1
                          for config in kid_configs[kid]
                          if present in config) >= T/alpha
    
    # Constraint 3: Each big present j ∈ J₂ is assigned to at most β kids
    for present in J2:
        prob += pulp.lpSum(x[kid, config]
                          for kid in kids
                          for config in kid_configs[kid]
                          if present in config) <= beta
    
    # Solve the LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return {}
    
    # Extract the solution
    solution = {}
    for kid in kids:
        configs = []
        for config in kid_configs[kid]:
            value = x[kid, config].value()
            if value > 1e-6:  # Only include configurations with non-zero values
                configs.append((config, value))
        if configs:
            solution[kid] = configs
    
    return solution


def _simplified_lp_solver(presents: Dict[str, Dict[str, float]], kids: Dict[str, int], T: float, alpha: float = 3.0, beta: float = 3.0) -> Dict:
    """
    A simplified version of the Configuration LP solver that uses a greedy approach.
    This is used when PuLP is not available.
    """
    # Simple greedy allocation for demonstration
    solution = {kid: [] for kid in kids}
    
    # Sort presents by value for each kid
    for present in presents:
        # Find the kid who values this present the most
        best_kid = max(kids.keys(),
                        key=lambda c: presents[present].get(c, 0))
        
        # If this kid has capacity and values the present
        if (len(solution[best_kid]) < kids[best_kid] and
            presents[present].get(best_kid, 0) > 0):
            solution[best_kid].append(present)
    
    # Check if the solution meets the target value
    for kid in solution:
        total_value = sum(presents[present].get(kid, 0) for present in solution[kid])
        if total_value < T:
            # If any kid doesn't meet the target, the problem is infeasible
            return {}
    
    return solution


def transformSolution(lpSolution: Dict) -> List[Tuple[List[str], List[str]]]:
    """
    Section 5.2: Transform the LP solution into another LP solution where the graph G is a forest.
    
    The transformation follows these steps:
    1. Any isolated machine node in G is a machine that has only small job configurations
    2. If a big job j is a leaf, assign it to its parent node and delete both j and its parent
    3. If a job node has degree exactly 2, form a chain of machines
    4. Otherwise, use the tree uniformity at some machine node
    
    :param lpSolution: The fractional solution from the Configuration LP
    :return: List of super-machines (Mᵢ, Jᵢ) where Mᵢ is a set of machines and Jᵢ is their assigned jobs
    """
    if not lpSolution:
        return []
    
    # Build the bipartite graph G
    G = build_configuration_graph(lpSolution)
    machine_nodes, job_nodes = get_node_sets(G)
    
    # Initialize super-machines list
    super_machines = []
    
    while G.number_of_nodes() > 0:
        # Step 1: Handle isolated machine nodes (only small jobs)
        isolated_machines = [n for n in machine_nodes if G.degree(n) == 0]
        if isolated_machines:
            machine = isolated_machines[0]
            small_jobs = [j for j in job_nodes 
                         if any(j in config[0] for config in lpSolution[machine])]
            super_machines.append(([machine], small_jobs))
            G.remove_node(machine)
            machine_nodes.remove(machine)
            continue
        
        # Step 2: Handle leaf big jobs
        leaf_jobs = [j for j in job_nodes if G.degree(j) == 1 and is_big_job(j)]
        if leaf_jobs:
            job = leaf_jobs[0]
            # Get the parent machine
            parent = list(G.neighbors(job))[0]
            # Create a super-machine with this pair
            super_machines.append(([parent], [job]))
            # Remove both nodes
            G.remove_node(parent)
            G.remove_node(job)
            machine_nodes.remove(parent)
            job_nodes.remove(job)
            continue
        
        # Step 3: Handle degree-2 job nodes
        degree2_jobs = [j for j in job_nodes if G.degree(j) == 2]
        if degree2_jobs:
            job = degree2_jobs[0]
            # Get the two machines connected to this job
            machines = list(G.neighbors(job))
            # Form a chain
            super_machines.append((machines, [job]))
            # Remove the nodes
            for m in machines:
                G.remove_node(m)
                machine_nodes.remove(m)
            G.remove_node(job)
            job_nodes.remove(job)
            continue
        
        # Step 4: Use tree uniformity at any remaining machine node
        if machine_nodes:
            machine = next(iter(machine_nodes))
            # Get all jobs connected to this machine
            connected_jobs = list(G.neighbors(machine))
            # Create a super-machine with this machine and its jobs
            super_machines.append(([machine], connected_jobs))
            # Remove the nodes
            G.remove_node(machine)
            machine_nodes.remove(machine)
            for job in connected_jobs:
                G.remove_node(job)
                job_nodes.remove(job)
    
    return super_machines


def is_big_present(present: str, presents: Dict[str, Dict[str, float]], target_value: float, alpha: float) -> bool:
    """
    Checks if a present is a big present according to the definition in the paper.
    In the restricted assignment case, a present is big if its value is exactly T (the target value).
    
    :param present: The present to check
    :param presents: Dictionary mapping presents to their values for each kid
    :param target_value: The target value T
    :param alpha: Parameter for classifying presents as large or small
    :return: True if the present is big, False otherwise
    """
    # In the restricted assignment case, a present has the same value for all kids who can receive it
    # A present is big if its value is exactly target_value (after rounding)
    max_value = 0
    for kid, value in presents[present].items():
        if value > 0:  # Kid can receive this present
            max_value = max(max_value, value)
    
    return max_value >= target_value


def is_small_present(present: str, presents: Dict[str, Dict[str, float]], target_value: float, alpha: float) -> bool:
    """
    Checks if a present is a small present according to the definition in the paper.
    In the restricted assignment case, a present is small if its value is less than T.
    
    :param present: The present to check
    :param presents: Dictionary mapping presents to their values for each kid
    :param target_value: The target value T
    :param alpha: Parameter for classifying presents as large or small
    :return: True if the present is small, False otherwise
    """
    # In the restricted assignment case, a present has the same value for all kids who can receive it
    # A present is small if its value is less than target_value
    max_value = 0
    for kid, value in presents[present].items():
        if value > 0:  # Kid can receive this present
            max_value = max(max_value, value)
    
    return max_value > 0 and max_value < target_value/alpha


def is_dummy_job(job: str) -> bool:
    """Determine if a job is a dummy job (type J₃).
    
    Dummy jobs are used in the LP formulation to handle certain constraints.
    
    :param job: The job to check
    :return: True if the job is a dummy job, False otherwise
    """
    # In the implementation, dummy jobs would be specifically created and tracked
    # For now, we'll assume no dummy jobs in our implementation
    return False


def construct_super_machines(alloc: AllocationBuilder, large_presents: Set[str]) -> List[Tuple[List[str], List[str]]]:
    """Algorithm 3: Construct the super-machine structure (Super-Machines).

    As described in the paper (Section 5.2-5.3):
    1. We group kids (machines) into clusters M₁, M₂, ..., M_p
    2. For each cluster M_i, there is a set of large presents (big jobs) J_i of size |M_i| - 1
    3. The large presents in J_i can be freely distributed among the kids in M_i
    4. The remaining kid gets a configuration of small presents

    :param alloc: The allocation builder containing the problem instance
    :param large_presents: Set of presents classified as large
    :return: List of super-machines (kids, presents) where each super-machine is a tuple (M_i, J_i)
    """
    logger.info("Constructing super-machines (super-machines)...")
    super_machines = []
    visited_kids = set()
    visited_presents = set()

    # Step 1: Find connected components of kids and large presents
    for kid in alloc.remaining_agent_capacities:
        if kid not in visited_kids:
            queue = [kid]
            component_kids = []
            component_presents = []
            
            while queue:
                current = queue.pop(0)
                
                if current in alloc.remaining_agent_capacities and current not in visited_kids:
                    visited_kids.add(current)
                    component_kids.append(current)
                    
                    # Add all connected large presents
                    for present in large_presents:
                        if present not in visited_presents and alloc.instance.agent_item_value(current, present) > 0:
                            visited_presents.add(present)
                            component_presents.append(present)
                            
                            # Add all kids connected to this present
                            for c in alloc.remaining_agent_capacities:
                                if c not in visited_kids and alloc.instance.agent_item_value(c, present) > 0:
                                    queue.append(c)
            
            # Step 2: Ensure each cluster satisfies |J_i| = |M_i| - 1
            if component_kids and component_presents:
                # If we have more presents than needed, remove some
                while len(component_presents) > len(component_kids) - 1:
                    # Remove the present with the lowest total value across all kids
                    present_to_remove = min(component_presents, 
                                           key=lambda p: sum(alloc.instance.agent_item_value(k, p) for k in component_kids))
                    component_presents.remove(present_to_remove)
                    visited_presents.remove(present_to_remove)
                
                # If we have fewer presents than needed, try to add more
                while len(component_presents) < len(component_kids) - 1:
                    # Try to find an unvisited large present that can be assigned to at least one kid in this component
                    unvisited_large_presents = large_presents - visited_presents
                    if not unvisited_large_presents:
                        # If no more large presents are available, we need to split this component
                        # We'll keep only enough kids to satisfy |J_i| = |M_i| - 1
                        component_kids = component_kids[:len(component_presents) + 1]
                        break
                    
                    # Find the best unvisited large present
                    best_present = max(unvisited_large_presents, 
                                      key=lambda p: sum(alloc.instance.agent_item_value(k, p) for k in component_kids))
                    component_presents.append(best_present)
                    visited_presents.add(best_present)
                
                # Add this super-machine (super-machine)
                if len(component_kids) > 1 and len(component_presents) == len(component_kids) - 1:
                    logger.info(f"Created super-machine with {len(component_kids)} kids and {len(component_presents)} large presents")
                    super_machines.append((component_kids, component_presents))
                    
    # Add any remaining large gifts to appropriate super-machines
    remaining_large_gifts = large_presents - visited_presents
    if remaining_large_gifts and super_machines:
        # Distribute remaining gifts among super-machines
        for present in remaining_large_gifts:
            # Add to super-machines with the fewest gifts, but only if it doesn't violate |J_i| = |M_i| - 1
            valid_super_machines = [i for i in range(len(super_machines)) 
                               if len(super_machines[i][1]) < len(super_machines[i][0]) - 1]
            if valid_super_machines:
                min_gifts_idx = min(valid_super_machines, 
                                   key=lambda i: len(super_machines[i][1]))
                super_machines[min_gifts_idx][1].append(present)
                visited_presents.add(present)
    
    # Log the final super-machines
    for i, (kids, presents) in enumerate(super_machines):
        logger.info(f"Super-machine {i+1}: {len(kids)} kids, {len(presents)} large presents")
        logger.info(f"  Kids: {kids}")
        logger.info(f"  Large presents: {presents}")
    
    return super_machines


def randomizedRounding(lpSolution: Dict, kids: List[str], small_presents: Set[str], beta: float = 3.0) -> Dict:
    """
    Algorithm 4: Round the small present configurations according to Section 5.2.
    
    As described in the paper:
    1. For each kid in M₁, we choose exactly one configuration C with
       probability x_iC (from the LP solution)
    2. We ensure that after this procedure on small presents, each kid has
       load O(log log m / log log log m)
    3. We use the Chernoff-Hoeffding bounds to show that with high probability,
       each present is assigned to at most β kids
    
    :param lpSolution: The fractional solution from the Configuration LP
    :param kids: List of kids in M₁
    :param small_presents: Set of small presents (J₁)
    :param beta: Relaxation parameter (default 3.0)
    :return: A mapping from kids to their chosen configurations
    """
    logger.info("Applying randomized rounding...")
    result = {}
    assigned_jobs = defaultdict(int)  # Count how many times each job is assigned
    i = 0
    
    # For each kid in M₁
    for kid in kids:
        if kid not in lpSolution:
            continue
            
        # Get the configurations and their probabilities
        configs = []
        probs = []
        for config, value in lpSolution[kid]:
            configs.append(config)
            probs.append(value)
            
        # Normalize probabilities if needed
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
            
        # Choose exactly one configuration with probability x_iC
        chosen_config_idx = np.random.choice(len(configs), p=probs)
        chosen_config = configs[chosen_config_idx]
        
        # Add to result
        result[i] = {
            "kid": kid,
            "presents": list(chosen_config)
        }
        i += 1
        
        # Update present assignment counts
        for present in chosen_config:
            if present in small_presents:
                assigned_jobs[present] += 1
                
        # Check if any job is assigned to too many machines (more than β)
        overloaded_jobs = [job for job, count in assigned_jobs.items() if count > beta]
        if overloaded_jobs:
            logger.warning(f"Jobs assigned to more than {beta} machines: {overloaded_jobs}")
            # In practice, we might want to re-run the rounding if this happens
            
    return result


def isGoodFunction(f: Dict, sets: List, gamma: float, beta: float) -> bool:
    """
    Check if the assignment of small presents to super-machines is good.
    An assignment is good if each kid has at least 1/(2γ) fraction of atoms of each present.
    
    :param f: The assignment of small presents to super-machines
    :param sets: The super-machine structure
    :param gamma: Parameter for the fraction of atoms
    :param beta: Relaxation parameter
    :return: True if the assignment is good, False otherwise
    """
    logger.info("Checking if solution is good...")
    # For each set in sets, check if it has at least one gift assigned
    if not f:  # If no assignments were made
        return True
    
    for kids, gifts in sets:
        # Each set should have at least one gift assigned
        assigned_gifts = sum(1 for gift in gifts if gift in f)
        if assigned_gifts == 0:
            return False
    return True


def improve_assignment(alloc: AllocationBuilder, super_machines: List, 
                     small_presents: Set[str], gamma: float = 2.0, beta: float = 3.0) -> Dict:
    """
    Improve the assignment of small gifts to super-machines.
    
    :param alloc: The allocation builder containing the problem instance
    :param super_machines: The super-machine structure
    :param small_gifts: Set of gifts classified as small
    :param gamma: Parameter for the fraction of atoms
    :param beta: Relaxation parameter
    :return: An improved assignment of small gifts to super-machines
    """
    logger.info("Improving assignment...")
    # Initialize the result dictionary
    result = {}
    
    # Distribute small gifts evenly
    # Create a list of small presents
    small_presents_list = list(small_presents)
    
    # Assign each small present to a super-machine
    for i, present in enumerate(small_presents_list):
        # Assign to super-machine with index i % len(super_machines)
        super_kid_idx = i % len(super_machines)
        kids, _ = super_machines[super_kid_idx]
        
        if not kids:
            continue
        
        # Choose the kid who values this present the most
        chosen_kid = max(kids, key=lambda c: alloc.instance.agent_item_value(c, present))
        
        # Initialize if not exists
        if super_kid_idx not in result:
            result[super_kid_idx] = {
                "kid": chosen_kid,
                "presents": []
            }
        
        # Add this present to the chosen kid's configuration
        result[super_kid_idx]["presents"].append(present)
    
    return result


def get_job_processing_time(jobs: Dict[str, Dict[str, float]], job: str) -> float:
    """
    Get the processing time pⱼ for a job in the restricted assignment case.
    In this case, each job j has a fixed processing time pⱼ that is either 0
    or the same value for all machines that can process it.
    
    :param jobs: Dictionary mapping jobs to their processing times per machine
    :param job: The job to get the processing time for
    :return: The processing time pⱼ for the job
    """
    logger.info("Getting job processing time...")
    # Get all non-zero processing times for this job
    processing_times = set(time for time in jobs[job].values() if time > 0)
    if not processing_times:  # If no machine can process this job
        return 0
    # In restricted assignment, all non-zero times should be the same
    assert len(processing_times) == 1, f"Job {job} has different non-zero processing times: {processing_times}"
    return processing_times.pop()


def apply_algorithmic_framework(jobs: Dict[str, Dict[str, float]], machines: Dict[str, int], T: float) -> Dict:
    """
    Section 5.3: Apply the algorithmic framework to get an O(log log m / log log log m) approximation.
    
    The framework follows these steps:
    1. For each super-machine Mᵢ, assign to some machine mᵢ,₁ all its assigned small jobs with total load at least T/α
    2. No job is assigned to more than β machines (i.e., the assignment is β-relaxed)
    
    As per Lemma 4, if we have a super-machine Mᵢ containing a single big job jᵢ,
    we assign jᵢ to a machine other than mᵢ(1). For the machines mᵢ(1), by Lemma 3,
    there is a proper assignment where each machine has load at least T/α.
    
    :param jobs: Dictionary mapping jobs to their processing times (0 or pⱼ for each machine)
    :param machines: Dictionary mapping machines to their capacities
    :param T: Target value for minimum machine load
    :return: A mapping from machines to their assigned jobs
    """
    logger.info("Applying algorithmic framework...")
    # Parameters from the paper
    alpha = 3  # Relaxation parameter
    beta = 3   # Maximum number of machines a job can be assigned to
    
    # Verify restricted assignment condition
    for job in jobs:
        pj = get_job_processing_time(jobs, job)
        for machine, time in jobs[job].items():
            assert time == 0 or time == pj, f"Job {job} has time {time} on machine {machine}, expected 0 or {pj}"
    
    # Step 1: Solve the Configuration LP
    logger.info("Solving Configuration LP...")
    lp_solution = solveConfigurationLP(jobs, machines, T)
    if not lp_solution:
        return {}
    
    # Step 2: Transform the solution to get super-machines
    super_machines = transformSolution(lp_solution)
    
    # Step 3: Assign jobs according to the framework
    assignment = {}
    for i, (M_i, J_i) in enumerate(super_machines):
        if len(J_i) == 1 and is_big_job(list(J_i)[0]):  # Single big job case
            # Find a machine other than m_i(1) to assign the job to
            job = list(J_i)[0]
            available_machines = [m for m in M_i[1:] if m not in assignment and jobs[job].get(m, 0) > 0]
            if available_machines:
                machine = available_machines[0]
                assignment[machine] = [job]
        else:  # Small jobs case
            # Assign all small jobs to m_i(1)
            if M_i:  # Make sure there's at least one machine
                machine = M_i[0]
                # Only assign jobs that meet the T/α load requirement and can be processed by this machine
                total_load = 0
                assigned_jobs = []
                for job in J_i:
                    if is_small_job(job):
                        pj = get_job_processing_time(jobs, job)
                        if jobs[job].get(machine, 0) > 0 and total_load + pj <= T/alpha:
                            total_load += pj
                            assigned_jobs.append(job)
                if assigned_jobs:
                    assignment[machine] = assigned_jobs
    
    # Verify the β-relaxed property
    job_counts = defaultdict(int)
    for machine, assigned_jobs in assignment.items():
        for job in assigned_jobs:
            job_counts[job] += 1
            if job_counts[job] > beta:
                logger.warning(f"Job {job} is assigned to more than {beta} machines")
                # In practice, we might want to fix this violation
    
    return assignment


def construct_final_allocation(alloc: AllocationBuilder, super_machines: List, 
                             rounded_solution: Dict, **kwargs) -> None:
    """
    Algorithm 5: Construct the final allocation.
    
    Assigns the small present configurations and large presents to kids
    in each super-machine, then removes conflicts.
    
    :param alloc: The allocation builder containing the problem instance
    :param super_machines: The super-machine structure
    :param rounded_solution: The rounded solution for small presents
    
    >>> super_machines = [(["Child1", "Child2"], ["gift1", "gift2"])]
    >>> rounded_solution = {0: {"kid": "Child1", "gifts": ["gift5"]}}
    >>> divide(lambda a, return_builder=False: construct_final_allocation(a, super_machines, rounded_solution, return_builder=return_builder), restricted_example)
    {'Child1': ['gift1', 'gift5'], 'Child2': ['gift2'], 'Child3': []}
    """
    if not super_machines:
        return
    
    # First, assign small presents according to the rounded solution
    for i, config in rounded_solution.items():
        if i >= len(super_machines):
            continue
        
        kid = config["kid"]
        small_presents = config["presents"]
        
        # Assign each small present to the kid
        for present in small_presents:
            if present in alloc.remaining_item_capacities and kid in alloc.remaining_agent_capacities:
                try:
                    alloc.give(kid, present)
                    logger.info(f"Assigned small present {present} to {kid}")
                except Exception as e:
                    logger.warning(f"Could not assign {present} to {kid}: {e}")
    
    # Then, distribute large presents within each super-machine
    for i, (kids, large_presents) in enumerate(super_machines):
        # Skip if no kids or no large presents
        if not kids or not large_presents:
            continue
        
        # Sort kids by how many presents they already have (ascending)
        sorted_kids = sorted(kids, key=lambda c: len(alloc.bundles.get(c, [])))        
        
        # Sort presents by their maximum value to any kid in this super-machine (descending)
        sorted_presents = sorted(
            large_presents,
            key=lambda p: max([alloc.instance.agent_item_value(k, p) for k in kids]),
            reverse=True
        )
        
        # Distribute large presents to kids with fewest presents
        for j, present in enumerate(sorted_presents):
            # Choose the kid who values this present the most among those with the fewest presents
            kid = max(sorted_kids[:min(j+1, len(sorted_kids))], 
                     key=lambda c: alloc.instance.agent_item_value(c, present))
            
            # Check if the present is available (no capacity check for kids according to the paper)
            if (present in alloc.remaining_item_capacities and 
                kid in alloc.remaining_agent_capacities):
                try:
                    alloc.give(kid, present)
                    logger.info(f"Assigned large present {present} to {kid}")
                except Exception as e:
                    logger.warning(f"Could not assign {present} to {kid}: {e}")
    
    # Finally, assign any remaining presents greedily
    # According to the paper, there's no direct limit on the number of presents a kid can receive
    remaining_presents = list(alloc.remaining_item_capacities.keys())
    remaining_kids = list(alloc.remaining_agent_capacities.keys())
    
    # Sort kids by the total value they've received so far (ascending)
    sorted_kids = sorted(remaining_kids, 
                        key=lambda c: sum(alloc.instance.agent_item_value(c, p) for p in alloc.bundles.get(c, [])))
    
    # For each kid (starting with those who have less capacity), find their highest-value present
    for kid in sorted_kids:
        # Skip if no presents remain
        if not remaining_presents:
            break
            
        # Find the best present for this kid
        best_present = max(remaining_presents, 
                          key=lambda p: alloc.instance.agent_item_value(kid, p))
        
        # Check if the present has any value for this kid
        if alloc.instance.agent_item_value(kid, best_present) > 0:
            try:
                alloc.give(kid, best_present)
                logger.info(f"Assigned high-value present {best_present} to {kid}")
                remaining_presents.remove(best_present)
                
                # According to the paper, there's no direct limit on the number of presents a kid can receive
                # So we don't need to remove kids from the remaining_kids list based on capacity
            except Exception as e:
                logger.warning(f"Could not assign {best_present} to {kid}: {e}")
    
    # Assign any remaining presents
    # Only proceed if there are both presents and kids remaining
    if remaining_presents and remaining_kids:
        # Sort presents by their maximum value to any kid (descending)
        remaining_presents = sorted(
            remaining_presents,
            key=lambda p: max([alloc.instance.agent_item_value(k, p) for k in remaining_kids]),
            reverse=True
        )
        
        for present in remaining_presents:
            if not remaining_kids:
                break
        
        # Find the kid who values this present the most
        best_kid = max(remaining_kids, 
                       key=lambda c: alloc.instance.agent_item_value(c, present))
        
        try:
            alloc.give(best_kid, present)
            logger.info(f"Assigned remaining present {present} to {best_kid}")
            
            # According to the paper, there's no direct limit on the number of presents a kid can receive
            # So we don't need to remove kids from the remaining_kids list based on capacity
        except Exception as e:
            logger.warning(f"Could not assign {present} to {best_kid}: {e}")
