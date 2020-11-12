import wikipedia

def main(name):
    imagelist = wikipedia.WikipediaPage(title=("Category:" + name)).images()
    print(imagelist)
    return

if __name__ == "__main__":
    main("Troides aeacus")