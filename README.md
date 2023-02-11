# YOLOv7-SBD

# To obtain drive access token

1. Visit this url with replaced params from oauth details:

    https://accounts.google.com/o/oauth2/auth?scope=https://www.googleapis.com/auth/drive&response_type=code&access_type=offline&redirect_uri=&client_id=

2. Authorize access.

3. Send the following curl request with new code given by GET request when redirected.

    curl -s --request POST --data "code=&client_id=&client_secret=&redirect_uri=&grant_type=authorization_code" https://accounts.google.com/o/oauth2/token
