import cv2
import numpy as np
import os
import glob
import random

def adjust_temperature(img, value):
        """Shift blue-red balance"""
        img = img.astype(np.float32)

        # Increase red, decrease blue for warm
        
        img[:, :, 2] += value  # Red
        img[:, :, 0] -= value  # Blue

        return np.clip(img, 0, 255).astype(np.uint8)


def adjust_lightness(img, value):
        """Brightness adjustment"""
        img = img.astype(np.float32)
        img += value
        return np.clip(img, 0, 255).astype(np.uint8)


def adjust_gamma(img, gamma):
        """Shadow/contrast control"""
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255 for i in range(256)
        ]).astype("uint8")

        return cv2.LUT(img, table)


def process_image(img, temp, light, gamma):
        out = adjust_temperature(img, temp)
        out = adjust_lightness(out, light)
        out = adjust_gamma(out, gamma)
        return out

splits = ["train","valid", "test"] #
for s in splits:
    # Independent controls
    temperature_values = [-70,0,70]  # negative = cooler, positive = warmer -40, -20, 0, 20, 40
    lightness_values = [-70,0,70]    # negative = darker, positive = brighter -50, -25, 0, 25, 50
    gamma_values = [0.8, 1.0, 1.2]              # shadow control (contrast-like) 0.8, 1.0, 1.2

    t_names = ["cooler", "normaltemp", "warmer"]
    l_names = ["darker", "normallight", "brighter"]

    input_path = f"./Covid LFAs/full_dataset/images/{s}/" #1650183370126.jpg  
   
    #process the images with all combinations of parameters
    images = glob.glob("{}/*.jpg".format(input_path))

    #for t in range(len(temperature_values)):
        #for l in range(len(lightness_values)):
            #for g in range(len(gamma_values)):

    
                #output_dir = "./Covid LFAs/lighting_conditions/{}/{}_{}/".format(s,t_names[t],l_names[l])
                #os.makedirs(output_dir, exist_ok=True)

    output_dir = "./Covid LFAs/lighting_conditions/random variation/images/{}/".format(s)
    os.makedirs(output_dir, exist_ok=True)

    for i in images: #tab if uncomment rows above until tabulation bellow matches
                    
                    temp = np.random.uniform(-70, 70)
                    light = np.random.uniform(-70, 70)
                    gamma = np.random.uniform(0.6, 1.3)

                    img = cv2.imread(i)
                    name = os.path.splitext(os.path.basename(i))[0]

                    results = []
                    labels = []
                    out = process_image(img, temp, light, gamma)

                    filename = f"{name}_T{temp}_L{light}_G{gamma}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), out)

                    results.append(out)
                    labels.append(filename)
    print("Done. Images saved in:", output_dir)


'''#uncomment for one image, different settings
        #plot differences

        cols = len(lightness_values)
        rows = len(temperature_values)

        h, w, _ = img.shape
        grid = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)

        i = 0
        for g in gamma_values:

            results = []

            for t in temperature_values:
                for l in lightness_values:
                    out = process_image(img, t, l, g)
                    results.append(out)

            rows = len(temperature_values)
            cols = len(lightness_values)

            h, w, _ = img.shape
            grid = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)

            i = 0
            for r in range(rows):
                for c in range(cols):
                    grid[r*h:(r+1)*h, c*w:(c+1)*w] = results[i]
                    i += 1

            cv2.imwrite(os.path.join(output_dir, f"grid_gamma_{g}.jpg"), grid)
'''

    