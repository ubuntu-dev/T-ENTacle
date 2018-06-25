import PyPDF2
import os
import shutil
from tika import parser

def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if 'doc' in input_pdf_path:
        shutil.copyfile(input_pdf_path, (target_dir + "/delete"))
        return

    with open(input_pdf_path, "rb") as input_stream:
        try:
            input_pdf = PyPDF2.PdfFileReader(input_stream)
        except PyPDF2.utils.PdfReadError:
            print("Could not read file: " + input_pdf_path + ". EOF marker not found.")

        if input_pdf.flattenedPages is None:
            # flatten the file using getNumPages()
            input_pdf.getNumPages()  # or call input_pdf._flatten()

        for num_page, page in enumerate(input_pdf.flattenedPages):
            output = PyPDF2.PdfFileWriter()
            output.addPage(page)

            file_name = os.path.join(target_dir, fname_fmt.format(num_page=num_page))
            with open(file_name, "wb") as output_stream:
                output.write(output_stream)


def remove_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def parse_document(file_path):
    """
    Calls tika to extract the text from a file per page and returns
    a list of strings, one per page
    TODO: change return from list to string for full file. Change all dependent code.
    TODO: make sure Tim Allison has fixed tika bug and we have all pulled new version of tika
    :param file_path: string. The path to the pdf to be parsed by tika
    :return parsed_text: list. Raw text extracted from the file
    """
    global current_document
    current_document=file_path.split('/')[-1]

    parsed_text=[]
    # create a dir for dumping split pdfs
    if os.path.exists('./temp'):
        shutil.rmtree('./temp/')
    else:
        os.mkdir('./temp')
    split_pdf_pages(file_path, 'temp')

    for pdf_page in os.listdir('temp'):
        # print('processing page: ',pdf_page)
        parsed = parser.from_file(os.path.join('temp', pdf_page))
        try:
            pdftext = parsed['content']
        except Exception:
            print("Could not read file.")
            pdftext=''

        parsed_text.append(pdftext)

    return parsed_text