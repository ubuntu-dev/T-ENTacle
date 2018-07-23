References:
https://github.com/USCDataScience/tika-dockers
https://wiki.apache.org/tika/ImageCaption
https://github.com/apache/tika/blob/master/tika-parsers/src/main/resources/org/apache/tika/parser/captioning/tf/im2txtapi.py

How to Run:
$ git clone --single-branch -b tablegen  \
https://github.com/nasa-jpl/T-ENTacle.git \
&& cd T-ENTacle/table-detection/dockers

$ docker build -f TFInferenceDockerfile -t tf-inference:v1.0 . --no-cache

$ docker run -p 9009:9009 -it tf-inference:v1.0
