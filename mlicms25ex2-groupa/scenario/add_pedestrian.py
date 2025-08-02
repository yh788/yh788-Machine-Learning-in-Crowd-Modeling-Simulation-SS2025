import json
import os
import subprocess

def add_pedestrian_to_scenario(input_file, output_file=None):
    """_summary_

    Args:
        input_file (str): Name of the input JSON scenario file to read.
        output_file (str, optional): Name of the output JSON scenario file to write. If None, overwrites the input file.

    Raises:
        ValueError: If the input file does not contain the expected scenario structure.

    Returns:
        _type_: True if pedestrian is added successfully, otherwise raises an error.
    """    
    # Check if output file name is provided, else use input file name
    if output_file is None:
        output_file = input_file
        

    # Read the JSON file
    with open(input_file, 'r') as f:
        scenario = json.load(f)
    
    # Create new pedestrian 
    new_pedestrian = {
        "attributes": {
            "id": 21,
            "shape": {
                "x": 0.0,
                "y": 0.0,
                "width": 1.0,
                "height": 1.0,
                "type": "RECTANGLE"
            },
            "visible": True,
            "radius": 0.2,
            "densityDependentSpeed": False,
            "speedDistributionMean": 1.34,
            "speedDistributionStandardDeviation": 0.26,
            "minimumSpeed": 0.5,
            "maximumSpeed": 2.2,
            "acceleration": 2.0,
            "footstepHistorySize": 4,
            "searchRadius": 1.0,
            "walkingDirectionSameIfAngleLessOrEqual": 45.0,
            "walkingDirectionCalculation": "BY_TARGET_CENTER"
        },
        "source": None,
        "targetIds": [100],
        "nextTargetListIndex": 0,
        "isCurrentTargetAnAgent": False,
        "position": {
            "x": 12.5,
            "y": 2.5
        },
        "velocity": {
            "x": 0.0,
            "y": 0.0
        },
        "freeFlowSpeed": 1.4,
        "followers": [],
        "idAsTarget": -1,
        "isChild": False,
        "isLikelyInjured": False,
        "psychologyStatus": {
            "mostImportantStimulus": None,
            "threatMemory": {
                "allThreats": [],
                "latestThreatUnhandled": False
            },
            "selfCategory": "TARGET_ORIENTED",
            "groupMembership": "OUT_GROUP",
            "knowledgeBase": {
                "knowledge": [],
                "informationState": "NO_INFORMATION"
            },
            "perceivedStimuli": [],
            "nextPerceivedStimuli": []
        },
        "healthStatus": None,
        "infectionStatus": None,
        "groupIds": [],
        "groupSizes": [],
        "agentsInGroup": [],
        "trajectory": {
            "footSteps": []
        },
        "modelPedestrianMap": None,
        "type": "PEDESTRIAN"
    }

    # Add the pedestrian to dynamicElements
    if "scenario" in scenario and "topography" in scenario["scenario"]:
        if "dynamicElements" not in scenario["scenario"]["topography"]:
            scenario["scenario"]["topography"]["dynamicElements"] = []
        
        scenario["scenario"]["topography"]["dynamicElements"].append(new_pedestrian)
        scenario["name"] = output_file
    else:
        raise ValueError("Invalid scenario structure: missing scenario.topography section")

    # Save the modified JSON
    with open(output_file, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    print(f"Successfully added pedestrian to {output_file}")
    return True

if __name__ == "__main__":
    # Define input and output files
    input = "RiMEA_Test6.scenario"
    output = "RiMEA_Test6_with_added_pedestrian.scenario"
    
    # Check if input file exists
    if not os.path.exists(input):
        print(f"File {input} doesn't exist.")
    else:
        # Add pedestrian to the scenario
        add_pedestrian_to_scenario(input, output)
        
    # Run Vadere simulation with console
    vadere_path = r"enter\path\to\vadere-console.jar" # Path to Vadere console jar fil
    scenario_path = os.path.abspath(output)  # Get full path to scenario file
    output_dir = r"enter\path/to\output" # Path to output directory for Vadere results


    vadere_cmd = [
        'java',
        '-jar',
        vadere_path,
        'scenario-run',
        '--scenario-file',
        scenario_path,
        '--output-dir',
        output_dir,
    ]
    
    try:
        result = subprocess.run(vadere_cmd, check=True, capture_output=True, text=True)
        print("Vadere simulation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running Vadere simulation: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")
