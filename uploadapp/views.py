from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .serializers import FileSerializer
import matplotlib.pyplot as plt
from PIL import Image

class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
      file_serializer = FileSerializer(data=request.data)
      if file_serializer.is_valid():
          try:
              print("Above")
              print(file_serializer.validated_data["file"])
              # print(file_serializer.validated_data["file"])
              # file_serializer.save()
              print("Below")
              return Response("file_serializer.data", status=status.HTTP_201_CREATED)
          except Exception as e:
              predictions = {"error": "2", "message": str(e)}
              return Response(predictions, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
