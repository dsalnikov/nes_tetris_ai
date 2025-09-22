import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import logging
import tetris_agent
import argparse
import pickle
import os


from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT

# Rotate Clockwise - a
# Rotate Counter-Clockwise  - b
# Drop - down
ACTION_MAPPING = {
    "NOOP": 0,
    "A": 1,
    "B": 2,
    "right": 3,
    "left": 4,
    "down": 5,
}


def extract_field(screen):
    img = np.ascontiguousarray(screen, dtype=np.uint8)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edge_detected = cv2.Canny(blurred_image, 25, 250)

    contours = cv2.findContours(edge_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    detected_screen = None
    for contour in sorted_contours:
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)

        if len(approximated_contour) == 4:
            detected_screen = approximated_contour
            break

    if detected_screen is not None:
        x,y,w,h = cv2.boundingRect(detected_screen)
        # add extra for border
        field = img[y:y+h, x:x+w]
        # get border width
        field = field[3:-3, 3:-3]

        boundary = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        cv2.drawContours(boundary, [detected_screen, ], -1, (0, 0, 255), 3)

        return field, blurred_image, edge_detected, boundary

    return None

def parse_field(field):
    field = cv2.cvtColor(field, cv2.COLOR_BGR2GRAY)
    field = np.ascontiguousarray(field, dtype=np.uint8)

    grid_size = field.shape[0] // 20, field.shape[1] // 10

    result = np.zeros((20, 10), dtype=np.uint8)

    for col in range(10):
        for row in range(20):
            x = grid_size[1] * col
            y = grid_size[0] * row
            h = grid_size[0]
            w = grid_size[1]
            roi = field[y:y+h, x:x+w]
            sum = np.sum(roi)
            result[row, col] = sum

    return result

def draw_debug_data(data, screen, field, result):
    blurred_image, edge_detected, boundary = data

    cv2.imshow("Input", cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    cv2.imshow("Gray & Blur", blurred_image)
    cv2.imshow("Canny", edge_detected)
    cv2.imshow("Boundary", boundary)

    cv2.imshow("Field", cv2.resize(cv2.cvtColor(field, cv2.COLOR_BGR2RGB), dsize=(200, 400), interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Result", cv2.resize(result * 100, dsize=(200, 400), interpolation=cv2.INTER_NEAREST))

def actions_to_string(actions):
    str_map = {
        'B': '↷',
        'left':"←",
        'right': "→",
        'down': "↓",
        'NOOP': ' ',
    }
    return "".join([str_map[a] for a in actions])

def move_to_actions(move):
    actions = []
    for rots in range(move[0]):
        actions.append("B")
        actions.append("NOOP")

    if move[1] > 0:
        for translations in range(move[1]):
            actions.append('right')
            actions.append('NOOP')
    else:
        for translations in range(abs(move[1])):
            actions.append('left')
            actions.append('NOOP')

    for i in range(10):
        actions.append('down')

    return actions

def create_initial_population(len):
    genomes = []
    for i in range(len):
        genomes.append({
            "rowsCleared": np.random.rand() - 0.5,
            "weightedHeight": np.random.rand() - 0.5,
            "cumulativeHeight": np.random.rand() - 0.5,
            "relativeHeight":np.random.rand() - 0.5,
            "holes":np.random.rand() - 0.5,
            "roughness":np.random.rand() - 0.5,
            "score": 0
        })
    return genomes

def play(env, args, genome):
    max_score = 0
    while True:
        state, reward, done, info = env.step(ACTION_MAPPING['NOOP'])
        if info['current_piece'] == None:
            continue

        if done:
            state = env.reset()
            return max_score

        if args.render:
            env.render()

        field, *debug_data = extract_field(state)
        result_field = parse_field(field)

        # draw_debug_data(debug_data, state, field, result_field)

        if result_field[0, 5] != 0:
            move = tetris_agent.get_move(result_field, info['current_piece'], genome)

            actions = move_to_actions(move)

            for action in actions:
                state, reward, done, info = env.step(ACTION_MAPPING[action])
                if done:
                    env.reset()
                    return max_score

            # number_of_lines
            max_score = max(max_score, info["score"])

        else:
            for action in ["down", "down", "down"]:
                state, reward, done, info = env.step(ACTION_MAPPING[action])
                if done:
                    env.reset()
                    return max_score

import random

def make_child(a, b):
    genome = {
        "rowsCleared": random.choice([a["rowsCleared"], b["rowsCleared"]]),
        "weightedHeight": random.choice([a["weightedHeight"], b["weightedHeight"]]),
        "cumulativeHeight": random.choice([a["cumulativeHeight"], b["cumulativeHeight"]]),
        "relativeHeight": random.choice([a["relativeHeight"], b["relativeHeight"]]),
        "holes": random.choice([a["holes"], b["holes"]]),
        "roughness": random.choice([a["roughness"], b["roughness"]])
    }
    mutationRate = 0.05
    mutationStep = 0.2
    for key in genome.keys():
        if np.random.rand() < mutationRate:
            genome[key] = genome[key] + np.random.rand() * mutationStep * 2 - mutationStep;

    genome["score"] = int(0)
    return genome



def evolve(genomes):
    print("evolve:")
    population_size = len(genomes)
    elites = sorted(genomes, key=lambda x : x['score'], reverse=True)[:population_size//2]

    genomes = []
    for i in range(population_size//2):
        genomes.append(make_child(random.choice(elites), random.choice(elites)))


    return elites + genomes

if __name__=="__main__":
    parser = argparse.ArgumentParser( prog='TetrisAI', description='AI playing Tetris')
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    else:
        logging.basicConfig(filename="logs/tetris_ai.log", format='%(levelname)s: %(message)s', level=logging.DEBUG)

    env = gym_tetris.make('TetrisA-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()

    population_size = 50
    genomes = create_initial_population(population_size)

    dumps = os.listdir("dumps")
    dumps = sorted(dumps)

    initial_generation = 0

    if len(dumps) > 0:
        initial_generation = int(dumps[-1].split(".")[0]) + 1
        with open(f"dumps/{dumps[-1]}", "rb") as f:
            print("loading genomes from dumps/{}".format(dumps[-1]))
            genomes = pickle.load(f)

    generations = initial_generation + 20

    for generation in range(initial_generation, generations):
        logging.info(f"generation: {generation}")

        for n, gene in enumerate(genomes):
            env.reset()
            score = play(env, args, gene)
            gene["score"] = int(score)
            logging.info(f"gene: {n} score: {score}")

        with open(f"dumps/{generation}.pickle", "wb") as f:
            pickle.dump(sorted(genomes, key=lambda x : x['score'], reverse=True), f)

        genomes = evolve(genomes)



    env.close()
