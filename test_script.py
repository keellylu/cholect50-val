from overeasy import *
from overeasy.models import OwlV2
from PIL import Image
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
import json
from ivtmetrics.recognition import Recognition
from ivtmetrics.detection import Detection
import os
import modal
import glob
import random

ANTHROPIC_API_KEY = "TODO"  # TODO: Replace with your own ANTHROPIC API key
GOOGLE_API_KEY = "TODO"  # TODO: Replace with your own GOOGLE API key
OPENAI_API_KEY = "TODO"  # TODO: Replace with your own OPENAI API key

overeasy_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update && apt-get install -y ffmpeg libsm6 libxext6")
    .pip_install("overeasy==0.1.6")
    .run_commands(
        "python -c 'import overeasy as ov; ov.models.warmup_models();'", gpu="any"
    )
    .pip_install("scikit-learn")
    .env({
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY
    })
)

tool_targets = {
    'grasper,dissect': {'omentum', 'gallbladder', 'cystic_plate'},
    'grasper,grasp': {'omentum', 'cystic_artery', 'gallbladder', 'cystic_duct', 'cystic_pedicle', 'liver', 'specimen_bag', 'cystic_plate', 'gut', 'peritoneum'},
    'grasper,pack': {'gallbladder'},
    'grasper,retract': {'omentum', 'gallbladder', 'cystic_duct', 'cystic_pedicle', 'liver', 'cystic_plate', 'gut', 'peritoneum'},
    'bipolar,coagulate': {'omentum', 'cystic_artery', 'gallbladder', 'cystic_duct', 'cystic_pedicle', 'cystic_plate', 'liver', 'blood_vessel', 'abdominal_wall_cavity', 'peritoneum'},
    'bipolar,dissect': {'omentum', 'cystic_artery', 'gallbladder', 'cystic_duct', 'adhesion', 'cystic_plate'},
    'bipolar,grasp': {'specimen_bag', 'liver', 'cystic_plate'},
    'bipolar,retract': {'omentum', 'gallbladder', 'cystic_duct', 'cystic_pedicle', 'liver'},
    'hook,coagulate': {'omentum', 'cystic_artery', 'gallbladder', 'cystic_duct', 'cystic_pedicle', 'liver', 'cystic_plate', 'blood_vessel'},
    'hook,cut': {'peritoneum', 'blood_vessel'},
    'hook,dissect': {'omentum', 'cystic_artery', 'gallbladder', 'cystic_duct', 'cystic_plate', 'blood_vessel', 'peritoneum'},
    'hook,retract': {'liver', 'gallbladder'},
    'scissors,coagulate': {'omentum'},
    'scissors,cut': {'omentum', 'cystic_artery', 'cystic_duct', 'liver', 'adhesion', 'cystic_plate', 'blood_vessel', 'peritoneum'},
    'scissors,dissect': {'omentum', 'gallbladder', 'cystic_plate'},
    'clipper,clip': {'cystic_artery', 'cystic_duct', 'cystic_pedicle', 'cystic_plate', 'blood_vessel'},
    'irrigator,aspirate': {'fluid'},
    'irrigator,dissect': {'omentum', 'gallbladder', 'cystic_duct', 'cystic_pedicle', 'cystic_plate'},
    'irrigator,irrigate': {'liver', 'cystic_pedicle', 'abdominal_wall_cavity'},
    'irrigator,retract': {'omentum', 'liver', 'gallbladder'},
    'grasper,null_verb': {'null_target'},
    'bipolar,null_verb': {'null_target'},
    'hook,null_verb': {'null_target'},
    'scissors,null_verb': {'null_target'},
    'clipper,null_verb': {'null_target'},
    'irrigator,null_verb': {'null_target'}
}

tool_verbs = {
    'grasper': {'pack', 'retract', 'dissect', 'null_verb', 'grasp'},
    'bipolar': {'retract', 'dissect', 'coagulate', 'null_verb', 'grasp'},
    'hook': {'retract', 'dissect', 'null_verb', 'coagulate', 'cut'},
    'scissors': {'dissect', 'null_verb', 'coagulate', 'cut'},
    'clipper': {'null_verb', 'clip'},
    'irrigator': {'retract', 'dissect', 'irrigate', 'null_verb', 'aspirate'}
}

class ToolType(Enum):
    GRASPER = "grasper"
    BIPOLAR = "bipolar"
    HOOK = "hook"
    SCISSORS = "scissors"
    CLIPPER = "clipper"
    IRRIGATOR = "irrigator"

class PhaseType(Enum):
    PREPARATION = "preparation"
    CARLOT_TRIANGLE_DISSECTION = "carlot triangle dissection"
    CLIPPING_AND_CUTTING = "clipping and cutting"
    GALLBLADDER_DISSECTION = "gallbladder dissection"
    GALLBLADDER_PACKAGING = "gallbladder packaging"
    CLEANING_AND_COAGULATION = "cleaning and coagulation"
    GALLBLADDER_EXTRACTION = "gallbladder extraction"

class TripletType(Enum):
    GRASPER_DISSECT_CYSTIC_PLATE = "grasper, dissect, cystic plate"
    GRASPER_DISSECT_GALLBLADDER = "grasper, dissect, gallbladder"
    GRASPER_DISSECT_OMENTUM = "grasper, dissect, omentum"
    GRASPER_GRASP_CYSTIC_ARTERY = "grasper, grasp, cystic artery"
    GRASPER_GRASP_CYSTIC_DUCT = "grasper, grasp, cystic duct"
    GRASPER_GRASP_CYSTIC_PEDICLE = "grasper, grasp, cystic pedicle"
    GRASPER_GRASP_CYSTIC_PLATE = "grasper, grasp, cystic plate"
    GRASPER_GRASP_GALLBLADDER = "grasper, grasp, gallbladder"
    GRASPER_GRASP_GUT = "grasper, grasp, gut"
    GRASPER_GRASP_LIVER = "grasper, grasp, liver"
    GRASPER_GRASP_OMENTUM = "grasper, grasp, omentum"
    GRASPER_GRASP_PERITONEUM = "grasper, grasp, peritoneum"
    GRASPER_GRASP_SPECIMEN_BAG = "grasper, grasp, specimen bag"
    GRASPER_PACK_GALLBLADDER = "grasper, pack, gallbladder"
    GRASPER_RETRACT_CYSTIC_DUCT = "grasper, retract, cystic duct"
    GRASPER_RETRACT_CYSTIC_PEDICLE = "grasper, retract, cystic pedicle"
    GRASPER_RETRACT_CYSTIC_PLATE = "grasper, retract, cystic plate"
    GRASPER_RETRACT_GALLBLADDER = "grasper, retract, gallbladder"
    GRASPER_RETRACT_GUT = "grasper, retract, gut"
    GRASPER_RETRACT_LIVER = "grasper, retract, liver"
    GRASPER_RETRACT_OMENTUM = "grasper, retract, omentum"
    GRASPER_RETRACT_PERITONEUM = "grasper, retract, peritoneum"
    BIPOLAR_COAGULATE_ABDOMINAL_WALL_CAVITY = "bipolar, coagulate, abdominal wall cavity"
    BIPOLAR_COAGULATE_BLOOD_VESSEL = "bipolar, coagulate, blood vessel"
    BIPOLAR_COAGULATE_CYSTIC_ARTERY = "bipolar, coagulate, cystic artery"
    BIPOLAR_COAGULATE_CYSTIC_DUCT = "bipolar, coagulate, cystic duct"
    BIPOLAR_COAGULATE_CYSTIC_PEDICLE = "bipolar, coagulate, cystic pedicle"
    BIPOLAR_COAGULATE_CYSTIC_PLATE = "bipolar, coagulate, cystic plate"
    BIPOLAR_COAGULATE_GALLBLADDER = "bipolar, coagulate, gallbladder"
    BIPOLAR_COAGULATE_LIVER = "bipolar, coagulate, liver"
    BIPOLAR_COAGULATE_OMENTUM = "bipolar, coagulate, omentum"
    BIPOLAR_COAGULATE_PERITONEUM = "bipolar, coagulate, peritoneum"
    BIPOLAR_DISSECT_ADHESION = "bipolar, dissect, adhesion"
    BIPOLAR_DISSECT_CYSTIC_ARTERY = "bipolar, dissect, cystic artery"
    BIPOLAR_DISSECT_CYSTIC_DUCT = "bipolar, dissect, cystic duct"
    BIPOLAR_DISSECT_CYSTIC_PLATE = "bipolar, dissect, cystic plate"
    BIPOLAR_DISSECT_GALLBLADDER = "bipolar, dissect, gallbladder"
    BIPOLAR_DISSECT_OMENTUM = "bipolar, dissect, omentum"
    BIPOLAR_GRASP_CYSTIC_PLATE = "bipolar, grasp, cystic plate"
    BIPOLAR_GRASP_LIVER = "bipolar, grasp, liver"
    BIPOLAR_GRASP_SPECIMEN_BAG = "bipolar, grasp, specimen bag"
    BIPOLAR_RETRACT_CYSTIC_DUCT = "bipolar, retract, cystic duct"
    BIPOLAR_RETRACT_CYSTIC_PEDICLE = "bipolar, retract, cystic pedicle"
    BIPOLAR_RETRACT_GALLBLADDER = "bipolar, retract, gallbladder"
    BIPOLAR_RETRACT_LIVER = "bipolar, retract, liver"
    BIPOLAR_RETRACT_OMENTUM = "bipolar, retract, omentum"
    HOOK_COAGULATE_BLOOD_VESSEL = "hook, coagulate, blood vessel"
    HOOK_COAGULATE_CYSTIC_ARTERY = "hook, coagulate, cystic artery"
    HOOK_COAGULATE_CYSTIC_DUCT = "hook, coagulate, cystic duct"
    HOOK_COAGULATE_CYSTIC_PEDICLE = "hook, coagulate, cystic pedicle"
    HOOK_COAGULATE_CYSTIC_PLATE = "hook, coagulate, cystic plate"
    HOOK_COAGULATE_GALLBLADDER = "hook, coagulate, gallbladder"
    HOOK_COAGULATE_LIVER = "hook, coagulate, liver"
    HOOK_COAGULATE_OMENTUM = "hook, coagulate, omentum"
    HOOK_CUT_BLOOD_VESSEL = "hook, cut, blood vessel"
    HOOK_CUT_PERITONEUM = "hook, cut, peritoneum"
    HOOK_DISSECT_BLOOD_VESSEL = "hook, dissect, blood vessel"
    HOOK_DISSECT_CYSTIC_ARTERY = "hook, dissect, cystic artery"
    HOOK_DISSECT_CYSTIC_DUCT = "hook, dissect, cystic duct"
    HOOK_DISSECT_CYSTIC_PLATE = "hook, dissect, cystic plate"
    HOOK_DISSECT_GALLBLADDER = "hook, dissect, gallbladder"
    HOOK_DISSECT_OMENTUM = "hook, dissect, omentum"
    HOOK_DISSECT_PERITONEUM = "hook, dissect, peritoneum"
    HOOK_RETRACT_GALLBLADDER = "hook, retract, gallbladder"
    HOOK_RETRACT_LIVER = "hook, retract, liver"
    SCISSORS_COAGULATE_OMENTUM = "scissors, coagulate, omentum"
    SCISSORS_CUT_ADHESION = "scissors, cut, adhesion"
    SCISSORS_CUT_BLOOD_VESSEL = "scissors, cut, blood vessel"
    SCISSORS_CUT_CYSTIC_ARTERY = "scissors, cut, cystic artery"
    SCISSORS_CUT_CYSTIC_DUCT = "scissors, cut, cystic duct"
    SCISSORS_CUT_CYSTIC_PLATE = "scissors, cut, cystic plate"
    SCISSORS_CUT_LIVER = "scissors, cut, liver"
    SCISSORS_CUT_OMENTUM = "scissors, cut, omentum"
    SCISSORS_CUT_PERITONEUM = "scissors, cut, peritoneum"
    SCISSORS_DISSECT_CYSTIC_PLATE = "scissors, dissect, cystic plate"
    SCISSORS_DISSECT_GALLBLADDER = "scissors, dissect, gallbladder"
    SCISSORS_DISSECT_OMENTUM = "scissors, dissect, omentum"
    CLIPPER_CLIP_BLOOD_VESSEL = "clipper, clip, blood vessel"
    CLIPPER_CLIP_CYSTIC_ARTERY = "clipper, clip, cystic artery"
    CLIPPER_CLIP_CYSTIC_DUCT = "clipper, clip, cystic duct"
    CLIPPER_CLIP_CYSTIC_PEDICLE = "clipper, clip, cystic pedicle"
    CLIPPER_CLIP_CYSTIC_PLATE = "clipper, clip, cystic plate"
    IRRIGATOR_ASPIRATE_FLUID = "irrigator, aspirate, fluid"
    IRRIGATOR_DISSECT_CYSTIC_DUCT = "irrigator, dissect, cystic duct"
    IRRIGATOR_DISSECT_CYSTIC_PEDICLE = "irrigator, dissect, cystic pedicle"
    IRRIGATOR_DISSECT_CYSTIC_PLATE = "irrigator, dissect, cystic plate"
    IRRIGATOR_DISSECT_GALLBLADDER = "irrigator, dissect, gallbladder"
    IRRIGATOR_DISSECT_OMENTUM = "irrigator, dissect, omentum"
    IRRIGATOR_IRRIGATE_ABDOMINAL_WALL_CAVITY = "irrigator, irrigate, abdominal wall cavity"
    IRRIGATOR_IRRIGATE_CYSTIC_PEDICLE = "irrigator, irrigate, cystic pedicle"
    IRRIGATOR_IRRIGATE_LIVER = "irrigator, irrigate, liver"
    IRRIGATOR_RETRACT_GALLBLADDER = "irrigator, retract, gallbladder"
    IRRIGATOR_RETRACT_LIVER = "irrigator, retract, liver"
    IRRIGATOR_RETRACT_OMENTUM = "irrigator, retract, omentum"
    GRASPER_NULL_VERB_NULL_TARGET = "grasper, null verb, null target"
    BIPOLAR_NULL_VERB_NULL_TARGET = "bipolar, null verb, null target"
    HOOK_NULL_VERB_NULL_TARGET = "hook, null verb, null target"
    SCISSORS_NULL_VERB_NULL_TARGET = "scissors, null verb, null target"
    CLIPPER_NULL_VERB_NULL_TARGET = "clipper, null verb, null target"
    IRRIGATOR_NULL_VERB_NULL_TARGET = "irrigator, null verb, null target"

class TargetType(Enum):
    GALLBLADDER = "gallbladder"
    CYSTIC_PLATE = "cystic plate"
    CYSTIC_DUCT = "cystic duct"
    CYSTIC_ARTERY = "cystic artery"
    CYSTIC_PEDICLE = "cystic pedicle"
    BLOOD_VESSEL = "blood vessel"
    FLUID = "fluid"
    ABDOMINAL_WALL_ACTIVITY = "abdominal wall activity"
    LIVER = "liver"
    ADHESION = "adhesion"
    OMENTUM = "omentum"
    PERITONEUM = "peritoneum"
    GUT = "gut"
    SPECIMEN_BAG = "specimen bag"
    NULL_TARGET = "other"

def get_tool_name(tool: ToolType):
    return tool.value

def get_phase_name(phase):
    return '_'.join(phase.value.split())

def get_triplet_name(triplet):
    return ','.join('_'.join(t.strip().split()) for t in triplet.value.split(','))

class Phase(BaseModel):
    phase: PhaseType = Field(description="Correctly identify the surgical phase that this photo displays.")

class Tool(BaseModel):
    tool: ToolType = Field(description="Correctly assign one of the predefined surgical tool types to the tool.")

class Triplet(BaseModel):
    triplet: TripletType = Field(description="Correctly identify the tool, verb, and target anatomy of that is going on in the operation.")

class Target(BaseModel):
    target: TargetType = Field(description="Correctly identify the target anatomy that the tool is operating")

def get_target_name(target):
    return '_'.join(target.value.split())

example_photos_dir = "example_photos"
app = modal.App(mounts=[
    modal.Mount.from_local_dir(local_path=example_photos_dir, remote_path=example_photos_dir)
])

@app.function(image=overeasy_image, concurrency_limit=40)
def modal_execute(image: Image.Image):
    import warnings
    import instructor
    import io, base64
    from overeasy.types import ImageAgent, ExecutionNode
    from openai import OpenAI
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    class InstructorImageAgent(ImageAgent):
        def __init__(self, response_model: type[BaseModel], model="gpt-4o"):
            self.model = model
            self.response_model = response_model

        def _encode_image(self, image: Image.Image) -> str:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        def _execute(self, image: Image.Image):
            client = instructor.from_openai(OpenAI())
            base64_image = self._encode_image(image)
            example_path = example_photos_dir + 'all/'
            example_photos = ["hook", "grasper", "scissors", "clipper", "bipolar", "irrigator"]
            msgs = []

            for photo in example_photos:
                encoded_buffers = []
                image_files = glob.glob(os.path.join(example_path, f"{photo}*.png"))
                selected_files = random.sample(image_files, 3)
                for image_file_path in selected_files:
                    with open(image_file_path, "rb") as image_file:
                        encoded_buffers.append(base64.b64encode(image_file.read()).decode('utf-8'))

                for buffer in encoded_buffers:
                    msgs.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{buffer}"}
                    })

                msgs.append({
                    "type": "text",
                    "text": f"These previous images are examples of a {photo} tool in a surgical setting, use this as a reference to label the target image."
                })

            # Extract structured data from natural language
            structured_response = client.chat.completions.create(
                model=self.model,
                response_model=self.response_model,
                messages=[{"role": "user", "content": [
                    *msgs,
                    {
                        "type": "text",
                        "text": """Classify the following target image.
                        Note that if the tool is picking up or manipulating a object it's likely a grasper
                        If it's cutting something it's likely either clippers or scissors.
                        If the tool doesn't have visible holes on it's shaft then it's not an irrigator.
                        If the tool has a white end and a small piece of metal sticking out its a hook
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                ]}],
            )

            return ExecutionNode(image, structured_response)

        def __repr__(self):
            model_name = self.model
            return f"{self.__class__.__name__}(response_model={self.response_model}, model={model_name})"

    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["medical tool", "surgical tool", "metal"], model=OwlV2()),
        NMSAgent(iou_threshold=0.1, score_threshold=0),
        PadCropAgent(x1_pad=70, y1_pad=70, x2_pad=70, y2_pad=70),
        SplitAgent(),
        InstructorImageAgent(Triplet),
        ToClassificationAgent(fn=lambda x: get_triplet_name(x.triplet)),
        JoinAgent(),
    ])

    try:
        result, _ = workflow.execute(image)
        return result
    except Exception as e:
        print(e)
        return []

@app.local_entrypoint()
def main():
    label_mappings_path = "cholect50-challenge-val/labels/VID75.json"
    with open(label_mappings_path, 'r') as file:
        label_mappings = json.load(file)
    triplet_mappings = {v: k for k, v in label_mappings["categories"]["triplet"].items()}

    vid75_path = "cholect50-challenge-val/labels/VID75.json"
    video_path = "cholect50-challenge-val/videos/VID75"

    with open(vid75_path, 'r') as file:
        vid75_mappings = json.load(file)

    v75_frame_mappings = vid75_mappings["annotations"]

    gt_classification_predictions = []
    classification_predictions = []
    images = []
    for filename in os.listdir(video_path):
        frame = filename[:-4].lstrip('0')

        if filename.endswith((".png")):
            image_path = os.path.join(video_path, filename)
            image = Image.open(image_path)
            images.append(image)

    correct_triplets = 0
    correct_tools = 0
    total = len(os.listdir(video_path))
    for filename, result in zip(os.listdir(video_path), modal_execute.map(images)):
        frame = filename[:-4].lstrip('0')

        if filename.endswith((".png")):
            predicted_triplet_names = []
            predicted_classes = np.zeros(100)
            triplet = result
            for i, bb in enumerate(triplet):
                triplet_name = triplet[i].data.classes[0]
                triplet_index = triplet_mappings[triplet_name]
                predicted_classes[int(triplet_index)] = 1
                predicted_triplet_names.append(triplet_name)

            classification_predictions.append(predicted_classes)
            print("File: ", filename)
            print("Predicted Triplets: ", predicted_triplet_names)

            # display ground truth detections
            correct_detections = []
            correct_classes = np.zeros(100)
            if frame in v75_frame_mappings:
                correct_detections = v75_frame_mappings[frame]
            correct_triplet_names: list[str] = []
            for detection in correct_detections:
                triplet_index = detection[0]
                correct_classes[int(triplet_index)] = 1
                triplet_name = label_mappings["categories"]["triplet"][str(triplet_index)]
                correct_triplet_names.append(triplet_name)

            gt_classification_predictions.append(correct_classes)
            print("Correct Triplets: ", correct_triplet_names)
            print("-------------------------------")

            if np.array_equal(correct_triplet_names, predicted_triplet_names):
                correct_triplets += 1

            if np.array_equal([x.split(',')[0] for x in correct_triplet_names], [x.split(',')[0] for x in predicted_triplet_names]):
                correct_tools += 1

    recognize = Recognition(num_class=100, ignore_null=False)
    recognize.update(gt_classification_predictions, classification_predictions)
    recognize.video_end()

    results_i = recognize.compute_video_AP('i')
    print("Tool Accuracy: ", correct_tools / total)
    print("Triplet Accuracy: ", correct_triplets / total)

    print("Instrument per class AP", results_i["AP"])
    print("Instrument mean AP", results_i["mAP"])

    results_it = recognize.compute_video_AP('it')
    print("Instrument-target mean AP", results_it["mAP"])

    results_ivt = recognize.compute_video_AP('ivt')
    print("Triplet mean AP", results_ivt["mAP"])