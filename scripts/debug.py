import pickle
import numpy as onp
import matplotlib.pyplot as plt

project_dir = "projects/2024_12_08/sapiens_envMerging-paths-alchemy_conn_dynamic_shared_batch_1_prob_visit_0.01_visit_dur_160_n_10_trial_0_lr_0.0001_rew_8"
#project_dir = "projects/2024_12_08/sapiens_envMerging-paths-alchemy_conn_dynamic_shared_batch_1_prob_visit_0.5_visit_dur_160_n_10_trial_0_lr_0.0001_rew_8"
project_dir = "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/projects/2024_12_09/sapiens_envMerging-paths-alchemy_conn_dynamic_shared_batch_1_prob_visit_0.01_visit_dur_320_n_10_trial_0_lr_0.0001_rew_8"
def check_neighbors():

    num_steps = 50000
    num_agents = 10
    neighbors = [onp.zeros((num_steps, 2)) for _ in range(num_agents)]
    for step in range(100,num_steps, 100):
        with open(project_dir + "/neighbors/step_" + str(step) + ".pkl", "rb") as f:
            step_neighbors = pickle.load(f)
            for row in range(step_neighbors.shape[0]):
                neighbors[row][step,...] = step_neighbors[row]

                print("agent " + str(row) + " has neighbors " + str(step_neighbors[row]))

    for agent in range(len(neighbors)):
        plt.plot(range(num_steps), neighbors[agent][...,0], label="neighbor_1")
        plt.plot(range(num_steps), neighbors[agent][...,1], label="neighbor_2")
        plt.savefig(project_dir + "/neighbors_agent_" + str(agent) + ".png")
        plt.clf()


def check_visiting():

    num_steps = 50000
    num_agents = 10
    neighbors = [onp.zeros((num_steps, )) for _ in range(num_agents)]
    for step in range(100,num_steps,100):
        with open(project_dir + "/visiting/step_" + str(step) + ".pkl", "rb") as f:
            step_neighbors = pickle.load(f)
            for row in range(step_neighbors.shape[0]):
                neighbors[row][step] = step_neighbors[row]


    for agent in range(len(neighbors)):
        plt.plot(range(num_steps), neighbors[agent], label="agent_" + str(agent))
        temp = onp.array(neighbors[agent])
        plt.savefig(project_dir + "/visiting_agent_" + str(agent) + ".png")
        plt.clf()


def check_metrics():

    num_steps = 2000
    num_agents = 10
    neighbors = [onp.zeros((num_steps, )) for _ in range(num_agents)]
    for step in range(num_steps, 100):
        with open(project_dir + "/metrics/step_" + str(step) + ".pkl", "rb") as f:
            step_neighbors = pickle.load(f)
            for row in range(step_neighbors.shape[0]):
                neighbors[row][step] = step_neighbors[row]


    for agent in range(len(neighbors)):
        plt.plot(range(num_steps), neighbors[agent], label="agent_" + str(agent))
        temp = onp.array(neighbors[agent])
    plt.savefig(project_dir + "/visiting_agent_" + str(agent) + ".png")
    plt.clf()

if __name__ == "__main__":
    check_neighbors()
    check_visiting()

    #check_metrics()