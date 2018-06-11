def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if 'doc' in input_pdf_path:
        shutil.copyfile(input_pdf_path, (target_dir + "/delete"))
        return

    with open(input_pdf_path, "rb") as input_stream:
        input_pdf = PyPDF2.PdfFileReader(input_stream)

        if input_pdf.flattenedPages is None:
            # flatten the file using getNumPages()
            input_pdf.getNumPages()  # or call input_pdf._flatten()

        for num_page, page in enumerate(input_pdf.flattenedPages):
            output = PyPDF2.PdfFileWriter()
            output.addPage(page)

            file_name = os.path.join(target_dir, fname_fmt.format(num_page=num_page))
            with open(file_name, "wb") as output_stream:
                output.write(output_stream)