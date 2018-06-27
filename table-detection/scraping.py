from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"fracking data tables with images study","limit":300,"color_type":"black-and-white", "print_urls":True,"size":"medium","chromedriver":"/path/to/chromedriver"}   

#creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)
