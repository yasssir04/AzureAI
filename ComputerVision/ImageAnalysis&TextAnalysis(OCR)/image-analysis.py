from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

# import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def main():

    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = 'images/qaz1.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]
        

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key))

        # Analyze image
        with open(image_file, "rb") as f:
            image_data = f.read()
        print(f'\nAnalyzing {image_file}\n')

        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE,
                VisualFeatures.READ],
        )


        # Get image captions
        if result.caption is not None:
            print("\nCaption:")
            print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))
    
        if result.dense_captions is not None:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

        # Get image tags
        if result.tags is not None:
            print("\nTags:")
            for tag in result.tags.list:
                print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))


        # Get objects in the image
        if result.objects is not None:
            print("\nObjects in image:")
            for detected_object in result.objects.list:
                # Print object tag and confidence
                print(" {} (confidence: {:.2f}%)".format(detected_object.tags[0].name, detected_object.tags[0].confidence * 100))
            # Annotate objects in the image
            show_objects(image_file, result.objects.list)


        # Get people in the image
        if result.people is not None:
            print("\nPeople in image:")

            for detected_person in result.people.list:
                if detected_person.confidence > 0.2:
                    # Print location and confidence of each person detected
                    print(" {} (confidence: {:.2f}%)".format(detected_person.bounding_box, detected_person.confidence * 100))
            # Annotate people in the image
            show_people(image_file, result.people.list)

            
        # Print the text
        if result.read is not None:
            print("\nText:")
    
            for line in result.read.blocks[0].lines:
                print(f" {line.text}")        
            # Annotate the text in the image
            annotate_lines(image_file, result.read)

            # Find individual words in each line
            print ("\nIndividual words:")
            for line in result.read.blocks[0].lines:
                for word in line.words:
                    print(f"  {word.text} (Confidence: {word.confidence:.2f}%)")
            # Annotate the words in the image
            annotate_words(image_file, result.read)
        
  
            
        
    except Exception as ex:
        print(ex)


def show_objects(image_file, detected_objects):
    print ("\nAnnotating objects...")

    # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_object in detected_objects:
        # Draw object bounding box
        r = detected_object.bounding_box
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height)) 
        draw.rectangle(bounding_box, outline=color, width=3)
        plt.annotate(detected_object.tags[0].name,(r.x, r.y), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    objectfile = 'objects.jpg'
    fig.savefig(objectfile)
    print('  Results saved in', objectfile)


def show_people(image_file, detected_people):
    print ("\nAnnotating objects...")

    # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_person in detected_people:
        if detected_person.confidence > 0.2:
            # Draw object bounding box
            r = detected_person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    peoplefile = 'people.jpg'
    fig.savefig(peoplefile)
    print('  Results saved in', peoplefile)


def annotate_lines(image_file, read_result):
    print ("\nAnnotating lines of text in image...")

    # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for line in read_result.blocks[0].lines:
        # Draw line bounding box
        r = line.bounding_polygon
        rectangle = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
        draw.polygon(rectangle, outline=color, width=3)
        #plt.annotate(line.text, (r.x, r.y), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    textfile = 'lines.jpg'
    fig.savefig(textfile)
    print('  Results saved in', textfile)

def annotate_words(image_file, read_result):
    print ("\nAnnotating individual  words in image...")

    # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for line in read_result.blocks[0].lines:
        for word in line.words:
            # Draw word bounding box
            r = word.bounding_polygon
            rectangle = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
            draw.polygon(rectangle, outline=color, width=3)
            #plt.annotate(word.text, (r.x, r.y), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    textfile = 'words.jpg'
    fig.savefig(textfile)
    print('  Results saved in', textfile)

if __name__ == "__main__":
    main()
