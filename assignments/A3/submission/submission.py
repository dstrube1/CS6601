# import sys

"""
WRITE YOUR CODE BELOW.
"""
# from numpy import zeros, float32
#  pgmpy
# import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Not allowed to use following set of modules from 'pgmpy' Library.
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

import random
import numpy as np

ALARM = "alarm"
FAULTY_ALARM = "faulty alarm"
GAUGE = "gauge"
FAULTY_GAUGE = "faulty gauge"
TEMPERATURE = "temperature"

SF = "SF"
MF = "MF"
OP = "OP"
MCP = "MCP"
RD = "RD"
LF = "LF"
GR = "GR"


def make_power_plant_net():
    """Create a Bayes Net representation of the power plant problem.
    Must use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature".
    (for the tests to work.)
    """
    bayes_net = BayesianModel()

    # Nodes
    # an alarm system will be going off or not
    bayes_net.add_node(ALARM)
    # the alarm system is broken or not
    bayes_net.add_node(FAULTY_ALARM)
    # the gauge will show either "above the threshold" or "below the threshold" (high = True, normal = False)
    bayes_net.add_node(GAUGE)
    # the gauge is broken
    bayes_net.add_node(FAULTY_GAUGE)
    # the temperature is HOT or NOT HOT (high = True, normal = False)
    bayes_net.add_node(TEMPERATURE)

    """
    T -> G
    T -> FG
    FG -> G
    G -> A
    FA -> A
    """

    # Connections
    # BayesNet.add_edge(<parent node name>,<child node name>)
    bayes_net.add_edge(TEMPERATURE, GAUGE)
    bayes_net.add_edge(TEMPERATURE, FAULTY_GAUGE)
    bayes_net.add_edge(FAULTY_GAUGE, GAUGE)
    bayes_net.add_edge(GAUGE, ALARM)
    bayes_net.add_edge(FAULTY_ALARM, ALARM)

    return bayes_net


def make_midterm_net():
    bayes_net = BayesianModel()

    # Nodes
    bayes_net.add_node(SF)
    bayes_net.add_node(MF)
    bayes_net.add_node(OP)
    bayes_net.add_node(MCP)
    bayes_net.add_node(RD)
    bayes_net.add_node(LF)
    bayes_net.add_node(GR)

    """
    SF -> MCP
    MF -> MCP
    OP -> RD
    MCP -> RD
    MCP -> LF
    RD -> GR
    LF -> GR
    """

    # Connections
    # BayesNet.add_edge(<parent node name>,<child node name>)
    bayes_net.add_edge(SF, MCP)
    bayes_net.add_edge(MF, MCP)
    bayes_net.add_edge(OP, RD)
    bayes_net.add_edge(MCP, RD)
    bayes_net.add_edge(MCP, LF)
    bayes_net.add_edge(RD, GR)
    bayes_net.add_edge(LF, GR)

    return bayes_net


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature".
    (for the tests to work.)
    """
    # CPD: Conditional probability distribution table
    # Seems a little weird to me to have the negatives before the positives, but whatever

    """
    P(TEMPERATURE): +: 0.2, -: 0.8
    P(FAULTY_ALARM): +: 0.15, -: 0.85
    P(FAULTY_GAUGE | TEMPERATURE): +: 0.8, -: 0.2 
    P(FAULTY_GAUGE | ! TEMPERATURE): +: 0.05, -: 0.95 
    P(GAUGE | TEMPERATURE): +: 0.2, -: 0.8 
    P(GAUGE| ! TEMPERATURE): +: 0.95, -: 0.05 
    P(GAUGE | FAULTY_GAUGE): +: 0.8, -: 0.2 
    P(GAUGE| ! FAULTY_GAUGE): +: 0.05, -: 0.95 
    P(ALARM | FAULTY_ALARM): +: 0.55, -: 0.45 
    P(ALARM | ! FAULTY_ALARM): +: 0.9, -: 0.1 
    P(ALARM | GAUGE): +: 0.45, -: 0.55 
    P(ALARM | ! GAUGE): +: 0.1, -: 0.9 
    """

    # P(A=true given G)
    # cpd_ag = TabularCPD('A', 2, values=[[0.15, 0.25], \
    #                     [ 0.85, 0.75]], evidence=['G'], evidence_card=[2])

    # P(A=true given G and T)
    # cpd_agt = TabularCPD('A', 2, values=[[0.9, 0.8, 0.4, 0.85], \
    #                     [0.1, 0.2, 0.6, 0.15]], evidence=['G', 'T'], evidence_card=[2, 2])

    # Handle invalid input
    if bayes_net is None:
        return bayes_net

    cpd_temperature = TabularCPD(TEMPERATURE, 2, values=[[0.8], [0.2]])
    cpd_faulty_alarm = TabularCPD(FAULTY_ALARM, 2, values=[[0.85], [0.15]])
    cpd_faulty_gauge = TabularCPD(FAULTY_GAUGE, 2, values=[[0.95, 0.2], [0.05, 0.8]], evidence=[TEMPERATURE],
                                  evidence_card=[2])
    cpd_gauge = TabularCPD(GAUGE, 2, values=[[0.95, 0.2, 0.05, 0.8], [0.05, 0.8, 0.95, 0.2]],
                           evidence=[TEMPERATURE, FAULTY_GAUGE], evidence_card=[2, 2])
    cpd_alarm = TabularCPD(ALARM, 2, values=[[0.9, 0.55, 0.1, 0.45], [0.1, 0.45, 0.9, 0.55]],
                           evidence=[GAUGE, FAULTY_ALARM], evidence_card=[2, 2])

    bayes_net.add_cpds(cpd_temperature, cpd_faulty_alarm, cpd_faulty_gauge, cpd_gauge, cpd_alarm)
    return bayes_net


def set_probability_midterm(bayes_net):
    """Set probability distribution for each node
    """
    # CPD: Conditional probability distribution table
    # Seems a little weird to me to have the negatives before the positives, but whatever

    """
    P(SF): +: 0.8, -: 0.2
    P(MF): +: 0.7, -: 0.3
    P(OP): +: 0.6, -: 0.4 
    
    P(RD | MCP, OP): +: 0.1, -: 0.9 
    P(RD | MCP, !OP): +: 0.5, -: 0.5 
    P(RD | !MCP, OP): +: 0.6, -: 0.4 
    P(RD | !MCP, !OP): +: 0.9, -: 0.1 
    
    P(MCP | SF, !MF): +: 0.35, -: 0.65 
    P(MCP | !SF, MF): +: 0.5, -: 0.5 
    P(MCP | !SF, !MF): +: 0.1, -: 0.9 
    
    P(LF | MCP): +: 0.1, -: 0.9 
    P(LF | !MCP): +: 0.6, -: 0.4 
    
    P(GR | LF, RD): +: 0.2, -: 0.8 
    P(GR | LF, !RD): +: 0.5, -: 0.5 
    P(GR | !LF, RD): +: 0.3, -: 0.7 
    P(GR | !LF, !RD): +: 0.8, -: 0.2 
    """

    # P(A=true given G)
    # cpd_ag = TabularCPD('A', 2, values=[[0.15, 0.25], \
    #                     [ 0.85, 0.75]], evidence=['G'], evidence_card=[2])

    # P(A=true given G and T)
    # cpd_agt = TabularCPD('A', 2, values=[[0.9, 0.8, 0.4, 0.85], \
    #                     [0.1, 0.2, 0.6, 0.15]], evidence=['G', 'T'], evidence_card=[2, 2])

    # Handle invalid input
    if bayes_net is None:
        return bayes_net

    cpd_SF = TabularCPD(SF, 2, values=[[0.2], [0.8]])
    cpd_MF = TabularCPD(MF, 2, values=[[0.3], [0.7]])
    cpd_OP = TabularCPD(OP, 2, values=[[0.4], [0.6]])

    cpd_RD = TabularCPD(RD, 2, values=[[0.1, 0.4, 0.5, 0.9], [0.9, 0.6, 0.5, 0.1]],
                        evidence=[MCP, OP], evidence_card=[2, 2])

    cpd_MCP = TabularCPD(MCP, 2, values=[[0.9, 0.5, 0.65, 0.1], [0.1, 0.5, 0.35, 0.9]],
                         evidence=[SF, MF], evidence_card=[2, 2])

    cpd_LF = TabularCPD(LF, 2, values=[[0.4, 0.9], [0.06, 0.1]], evidence=[MCP], evidence_card=[2])

    cpd_GR = TabularCPD(GR, 2, values=[[0.2, 0.7, 0.5, 0.8], [0.8, 0.3, 0.5, 0.2]],
                        evidence=[LF, RD], evidence_card=[2, 2])

    bayes_net.add_cpds(cpd_SF, cpd_MF, cpd_OP, cpd_RD,  # """cpd_MCP,"""
        cpd_LF, cpd_GR)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal probability of the alarm
    ringing in the power plant system."""

    # Handle invalid input
    if bayes_net is None:
        return None

    inference = VariableElimination(bayes_net)
    alarm_prob = inference.query(variables=[ALARM], joint=False)
    return alarm_prob[ALARM].values[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal probability of the gauge
    showing hot in the power plant system."""

    # Handle invalid input
    if bayes_net is None:
        return None

    inference = VariableElimination(bayes_net)
    gauge_prob = inference.query(variables=[GAUGE], joint=False)
    return gauge_prob[GAUGE].values[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability of the temperature being hot in the
    power plant system, given that the alarm sounds and neither the gauge
    nor alarm is faulty."""

    # Handle invalid input
    if bayes_net is None:
        return None

    inference = VariableElimination(bayes_net)
    temp_prob = inference.query(variables=[TEMPERATURE], evidence={ALARM: 1, FAULTY_ALARM: 0, FAULTY_GAUGE: 0},
                                joint=False)
    return temp_prob[TEMPERATURE].values[1]


def get_MCP_prob(bayes_net):
    """Calculate the conditional probability of MCP, given SF and MF."""

    # Handle invalid input
    if bayes_net is None:
        return None

    inference = VariableElimination(bayes_net)
    temp_prob = inference.query(variables=[MCP], evidence={SF: 1, MF: 1},
                                joint=False)
    return temp_prob[GR].values[1]


def get_GR_prob(bayes_net):
    """Calculate the conditional probability of GR, given MCP and OP."""

    # Handle invalid input
    if bayes_net is None:
        return None

    inference = VariableElimination(bayes_net)
    temp_prob = inference.query(variables=[GR], evidence={MCP: 1, OP: 1},
                                joint=False)
    return temp_prob[GR].values[1]


midterm_net = make_midterm_net()
midterm_net = set_probability_midterm(midterm_net)
# print("get_MCP_prob:")
# print(get_MCP_prob(midterm_net))
print("get_GR_prob:")
print(get_GR_prob(midterm_net))

AIRHEADS = "A"
BUFFOONS = "B"
CLODS = "C"
AIRHEADS_V_BUFFOONS = "AvB"
BUFFOONS_V_CLODS = "BvC"
CLODS_V_AIRHEADS = "CvA"
FIRST_WINS = 0
SECOND_WINS = 1
TIE = 2


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    # Nodes
    bayes_net = BayesianModel()
    bayes_net.add_node(AIRHEADS)
    bayes_net.add_node(BUFFOONS)
    bayes_net.add_node(CLODS)
    bayes_net.add_node(AIRHEADS_V_BUFFOONS)
    bayes_net.add_node(BUFFOONS_V_CLODS)
    bayes_net.add_node(CLODS_V_AIRHEADS)

    # Edges
    bayes_net.add_edge(AIRHEADS, AIRHEADS_V_BUFFOONS)
    bayes_net.add_edge(AIRHEADS, CLODS_V_AIRHEADS)
    bayes_net.add_edge(BUFFOONS, AIRHEADS_V_BUFFOONS)
    bayes_net.add_edge(BUFFOONS, BUFFOONS_V_CLODS)
    bayes_net.add_edge(CLODS, BUFFOONS_V_CLODS)
    bayes_net.add_edge(CLODS, CLODS_V_AIRHEADS)

    # Probabilities
    cpd_a = TabularCPD(AIRHEADS, 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_b = TabularCPD(BUFFOONS, 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_c = TabularCPD(CLODS, 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_avb = TabularCPD(AIRHEADS_V_BUFFOONS, 3, values=[
        [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
        [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    ], evidence=[AIRHEADS, BUFFOONS], evidence_card=[4, 4])
    cpd_bvc = TabularCPD(BUFFOONS_V_CLODS, 3, values=[
        [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
        [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    ], evidence=[BUFFOONS, CLODS], evidence_card=[4, 4])
    cpd_cva = TabularCPD(CLODS_V_AIRHEADS, 3, values=[
        [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
        [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    ], evidence=[CLODS, AIRHEADS], evidence_card=[4, 4])
    bayes_net.add_cpds(cpd_a, cpd_b, cpd_c, cpd_avb, cpd_bvc, cpd_cva)

    return bayes_net


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""

    # Handle invalid input
    if bayes_net is None:
        return None

    inference = VariableElimination(bayes_net)
    posterior = inference.query(variables=[BUFFOONS_V_CLODS], evidence={AIRHEADS_V_BUFFOONS: FIRST_WINS,
                                                                        CLODS_V_AIRHEADS: TIE},
                                joint=False)
    return posterior[BUFFOONS_V_CLODS].values


# noinspection PyPep8Naming
def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """

    # Handle invalid input
    if bayes_net is None:
        return None

    if initial_state is None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3),
            random.randint(0, 3),
            random.randint(0, 3),
            FIRST_WINS,
            random.randint(0, 2),
            TIE
        )
        return initial_state

    # pick variable to sample
    sample_index = random.randint(0, 5)

    # Sample the chosen variable
    sample = tuple(initial_state)

    # AIRHEADS
    if sample_index == 0:
        a_probs = bayes_net.get_cpds(AIRHEADS).values
        b_value = initial_state[1]
        c_value = initial_state[2]
        avb_probs = bayes_net.get_cpds(AIRHEADS_V_BUFFOONS).values[0]
        cva_probs = bayes_net.get_cpds(CLODS_V_AIRHEADS).values[2]

        likelihood_numerator_a = []
        # Find numerator for posterior calculation
        for i in range(len(a_probs)):
            numerator = a_probs[i] * avb_probs[i][b_value] * cva_probs[c_value][i]
            likelihood_numerator_a.append(numerator)

        # normalize numerators
        sum_a = sum(likelihood_numerator_a)
        likelihoods = np.array(likelihood_numerator_a) / sum_a
        # Randomly select the new value based on the given distribution
        new_a = np.random.choice([0, 1, 2, 3], p=likelihoods)
        sample = new_a, b_value, c_value, 0, initial_state[4], 2
    # BUFFOONS
    if sample_index == 1:
        b_probs = bayes_net.get_cpds(BUFFOONS).values
        a_value = initial_state[0]
        c_value = initial_state[2]
        avb_probs = bayes_net.get_cpds(AIRHEADS_V_BUFFOONS).values[0]
        bvc_probs = bayes_net.get_cpds(BUFFOONS_V_CLODS).values[initial_state[4]]

        likelihood_numerator_b = []
        # Find numerator for posterior calculation
        for i in range(len(b_probs)):
            numerator = b_probs[i] * avb_probs[a_value][i] * bvc_probs[i][c_value]
            likelihood_numerator_b.append(numerator)

        # normalize numerators
        sum_b = sum(likelihood_numerator_b)
        likelihoods = np.array(likelihood_numerator_b) / sum_b
        # Randomly select the new value based on the given distribution
        new_b = np.random.choice([0, 1, 2, 3], p=likelihoods)
        sample = a_value, new_b, c_value, 0, initial_state[4], 2
    # CLODS
    if sample_index == 2:
        c_probs = bayes_net.get_cpds(CLODS).values
        b_value = initial_state[1]
        a_value = initial_state[0]
        bvc_probs = bayes_net.get_cpds(AIRHEADS_V_BUFFOONS).values[initial_state[4]]
        cva_probs = bayes_net.get_cpds(CLODS_V_AIRHEADS).values[2]

        likelihood_numerator_c = []
        # Find numerator for posterior calculation
        for i in range(len(c_probs)):
            numerator = c_probs[i] * bvc_probs[b_value][i] * cva_probs[i][a_value]
            likelihood_numerator_c.append(numerator)

        # normalize numerators
        sum_c = sum(likelihood_numerator_c)
        likelihoods = np.array(likelihood_numerator_c) / sum_c
        # Randomly select the new value based on the given distribution
        new_c = np.random.choice([0, 1, 2, 3], p=likelihoods)
        sample = a_value, b_value, new_c, 0, initial_state[4], 2
    # AIRHEADS_V_BUFFOONS
    if sample_index == 3:
        avb_probs = bayes_net.get_cpds(AIRHEADS_V_BUFFOONS).values
        a_value = initial_state[1]
        b_value = initial_state[2]

        numerators = []
        for i in range(3):
            numerator = avb_probs[i][a_value][b_value]
            numerators.append(numerator)

        # normalize numerators
        sum_avb = sum(numerators)
        likelihoods = np.array(numerators) / sum_avb
        # Randomly select the new value based on the given distribution
        new_avb = np.random.choice([0, 1, 2], p=likelihoods)
        sample = initial_state[0], a_value, b_value, 0, new_avb, 2
    # BUFFOONS_V_CLODS
    if sample_index == 4:
        bvc_probs = bayes_net.get_cpds(BUFFOONS_V_CLODS).values
        b_value = initial_state[1]
        c_value = initial_state[2]

        numerators = []
        for i in range(3):
            numerator = bvc_probs[i][b_value][c_value]
            numerators.append(numerator)

        # normalize numerators
        sum_bvc = sum(numerators)
        likelihoods = np.array(numerators) / sum_bvc
        # Randomly select the new value based on the given distribution
        new_bvc = np.random.choice([0, 1, 2], p=likelihoods)
        sample = initial_state[0], b_value, c_value, 0, new_bvc, 2
    # CLODS_V_AIRHEADS
    if sample_index == 5:
        cva_probs = bayes_net.get_cpds(CLODS_V_AIRHEADS).values
        c_value = initial_state[1]
        a_value = initial_state[2]

        numerators = []
        for i in range(3):
            numerator = cva_probs[i][c_value][a_value]
            numerators.append(numerator)

        # normalize numerators
        sum_cva = sum(numerators)
        likelihoods = np.array(numerators) / sum_cva
        # Randomly select the new value based on the given distribution
        new_cva = np.random.choice([0, 1, 2], p=likelihoods)
        sample = initial_state[0], c_value, a_value, 0, new_cva, 2
    return sample


# noinspection PyPep8Naming
def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Metropolis-Hastings sampling algorithm given a Bayesian network and an
    initial state value.
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 

    A_cpd = bayes_net.get_cpds("A")
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)
    return sample"""

    # Handle invalid input
    if bayes_net is None:
        return None

    if initial_state is None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3),
            random.randint(0, 3),
            random.randint(0, 3),
            FIRST_WINS,
            random.randint(0, 2),
            TIE
        )
        return initial_state

    # Generate random walk
    new_a = random.randint(0, 3)
    new_b = random.randint(0, 3)
    new_c = random.randint(0, 3)
    new_avb = random.randint(0, 2)
    new_bvc = random.randint(0, 2)
    new_cva = random.randint(0, 2)
    candidate = (new_a, new_b, new_c, new_avb, new_bvc, new_cva)

    # Get cpds
    a_prob = bayes_net.get_cpds(AIRHEADS).values
    b_prob = bayes_net.get_cpds(BUFFOONS).values
    c_prob = bayes_net.get_cpds(CLODS).values
    avb_prob = bayes_net.get_cpds(AIRHEADS_V_BUFFOONS).values
    bvc_prob = bayes_net.get_cpds(BUFFOONS_V_CLODS).values
    cva_prob = bayes_net.get_cpds(CLODS_V_AIRHEADS).values

    # Calculate likelihood of candidate
    p_a = a_prob[new_a]
    p_b = b_prob[new_b]
    p_c = c_prob[new_c]
    p_avb = avb_prob[new_avb][new_a][new_b]
    p_bvc = bvc_prob[new_bvc][new_b][new_c]
    p_cva = cva_prob[new_cva][new_c][new_a]

    p_cand = p_a * p_b * p_c * p_avb * p_bvc * p_cva

    # Calculate the likelihood of the initial state
    p_a = a_prob[initial_state[0]]
    p_b = b_prob[initial_state[1]]
    p_c = c_prob[initial_state[2]]
    p_avb = avb_prob[new_avb][initial_state[0]][initial_state[1]]
    p_bvc = bvc_prob[initial_state[4]][initial_state[1]][initial_state[2]]
    p_cva = cva_prob[new_cva][initial_state[2]][initial_state[0]]

    p_initial = p_a * p_b * p_c * p_avb * p_bvc * p_cva

    # Accept or reject candidate
    alpha = min(1, p_cand / p_initial)
    acceptance_criterion = random.uniform(0, 1)
    if acceptance_criterion < alpha:
        sample = candidate
    else:
        sample = initial_state
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."" "
    gibbs_count = 0
    mh_count = 0
    mh_rejection_count = 0
    gibbs_convergence = [0, 0, 0]  # posterior distribution of the BvC match as produced by Gibbs
    mh_convergence = [0, 0, 0]  # posterior distribution of the BvC match as produced by MH
    return gibbs_convergence, mh_convergence, gibbs_count, mh_count, mh_rejection_count"""

    # Handle invalid input
    if bayes_net is None:
        return None

    gibbs_count = 0
    mh_count = 0
    mh_rejection_count = 0
    delta = 0.00001
    index_max = 100000

    # Calculate Gibbs
    current_dist = np.array([0, 0, 0])
    previous_dist = np.array([0, 0, 0])
    current_state = initial_state
    convergence_counter = 0
    convergence_counter_limit = 100
    for _ in range(index_max):
        new_state = Gibbs_sampler(bayes_net, current_state)
        current_state = new_state
        current_dist[new_state[4]] += 1

        # Normalize current and previous distributions to get probabilities
        current_normal = current_dist / np.sum(current_dist)
        previous_normal = previous_dist
        if np.sum(previous_dist) != 0:
            previous_normal = previous_dist / np.sum(previous_dist)
        diff = np.average(np.absolute(current_normal - previous_normal))
        if diff <= delta:
            convergence_counter += 1
            if convergence_counter == convergence_counter_limit:
                gibbs_count += 1
                break
        else:
            convergence_counter = 0
        previous_dist = np.copy(current_dist)
        gibbs_count += 1
    gibbs_convergence = current_dist / np.sum(current_dist)

    # Calculate MH
    current_dist = np.array([0, 0, 0])
    previous_dist = np.array([0, 0, 0])
    current_state = initial_state
    convergence_counter = 0
    for _ in range(index_max):
        candidate = MH_sampler(bayes_net, current_state)
        if candidate == current_state:
            mh_rejection_count += 1
        current_dist[candidate[4]] += 1
        current_state = candidate
        # Normalize current and previous distributions to get probabilities
        current_normal = current_dist / np.sum(current_dist)
        previous_normal = previous_dist
        if np.sum(previous_dist) != 0:
            previous_normal = previous_dist / np.sum(previous_dist)
        diff = np.average(np.absolute(current_normal - previous_normal))
        if diff <= delta:
            convergence_counter += 1
            if convergence_counter == convergence_counter_limit:
                mh_count += 1
                break
        else:
            convergence_counter = 0
        previous_dist = np.copy(current_dist)
        mh_count += 1
    mh_convergence = current_dist / np.sum(current_dist)

    return gibbs_convergence, mh_convergence, gibbs_count, mh_count, mh_rejection_count


def sampling_question():
    """Question about sampling performance."""
    game_network = get_game_network()
    gibbs_convergence, mh_convergence, gibbs_count, mh_count, mh_rejection_count = compare_sampling(game_network, [])

    # assign value to choice and factor
    if gibbs_count < mh_count:
        choice = 0
        factor = mh_count / gibbs_count
    else:
        choice = 1
        factor = gibbs_count / mh_count
    options = ['Gibbs', 'Metropolis-Hastings']
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return "David Strube"
