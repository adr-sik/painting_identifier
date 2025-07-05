import os
import image_identifier as img_id

def main():
    while True:
        query_img = input('Image path: ')
        results = img_id.find_similar_images(query_img)
        
        if query_img.lower() == "exit":
            print("Exiting the program.")
            break        
        if not os.path.isfile(query_img):
            print("File does not exist.")
            continue
            
        if results:
            for r in results:
                print(f"Author: {r['author']}, Painting: {r['painting']}")
        else:
            print('No similar images found.')

if __name__ == '__main__':
    main()