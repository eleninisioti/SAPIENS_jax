import pickle
import numpy as onp
import matplotlib.pyplot as plt

project_dir = "projects/2025_04_06/sapiens_envSingle-path-alchemy_conn_dynamic_shared_batch_1_prob_visit_0.1_visit_dur_5_n_10_lr_0.0001_rew_8/trial_0"

def check_neighbors():

    num_steps = 100
    num_agents = 10
    neighbors = [onp.zeros((num_steps, 10)) for _ in range(num_agents)]
    default_neighbors = [1,0,3,2,5,4,7,6,9,8]
    for step in range(1,num_steps):
        print(step)
        with open(project_dir + "/neighbors/step_" + str(step) + ".pkl", "rb") as f:
            step_neighbors = pickle.load(f)

            for row in range(step_neighbors.shape[0]):
                neighbors[row][step,...] = step_neighbors[row]
                print(step_neighbors[row])


                with open(project_dir + "/visit_log.txt", "a") as file:
                    print(step,file=file)

                    print("agent " + str(row) + " has neighbors " + str(step_neighbors[row]), file=file)

    for agent in range(len(neighbors)):
        plt.plot(range(num_steps), neighbors[agent][...,0], label="neighbor_0")
        plt.plot(range(num_steps), neighbors[agent][...,1], label="neighbor_1")
        plt.plot(range(num_steps), neighbors[agent][...,2], label="neighbor_2")
        plt.plot(range(num_steps), neighbors[agent][...,3], label="neighbor_3")
        plt.plot(range(num_steps), neighbors[agent][...,4], label="neighbor_4")
        plt.plot(range(num_steps), neighbors[agent][...,5], label="neighbor_5")
        plt.plot(range(num_steps), neighbors[agent][...,6], label="neighbor_6")
        plt.plot(range(num_steps), neighbors[agent][...,7], label="neighbor_7")
        plt.plot(range(num_steps), neighbors[agent][...,8], label="neighbor_8")
        plt.plot(range(num_steps), neighbors[agent][...,9], label="neighbor_9")
        plt.savefig(project_dir + "/neighbors_agent_" + str(agent) + ".png")
        plt.clf()


def check_visiting():

    num_steps = 100
    num_agents = 10
    neighbors = [[0] for _ in range(num_agents)]
    for step in range(1,num_steps):
        with open(project_dir + "/visiting/step_" + str(step) + ".pkl", "rb") as f:
            step_neighbors = pickle.load(f)
            for idx, el in enumerate(list(step_neighbors)):

                try:
                    neighbors[idx].append(el)
                except IndexError:
                    print("check")

    for agent in range(len(neighbors)):
        plt.plot(range(num_steps), neighbors[agent], label="agent_" + str(agent))
        temp = onp.array(neighbors[agent])
        plt.savefig(project_dir + "/visiting_agent_" + str(agent) + ".png")
        plt.clf()

def check_groups():

    num_steps = 100
    num_agents = 10
    neighbors = [[0] for _ in range(num_agents)]
    for step in range(1,num_steps):
        with open(project_dir + "/groups/step_" + str(step) + ".pkl", "rb") as f:
            step_neighbors = pickle.load(f)
            for idx, el in enumerate(list(step_neighbors)):

                try:
                    neighbors[idx].append(el)
                except IndexError:
                    print("check")

    for agent in range(len(neighbors)):
        plt.plot(range(num_steps), neighbors[agent], label="agent_" + str(agent))
        temp = onp.array(neighbors[agent])
        plt.savefig(project_dir + "/groups_agent_" + str(agent) + ".png")
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

    check_groups()

    #check_metrics()
