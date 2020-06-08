import json
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .serializers import FileSerializer
from predictionS2 import predictS2
from predictionS1 import predictS1
from predictionS1_S2 import predictS1_S2
class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
      file_serializer = FileSerializer(data=request.data)
      if file_serializer.is_valid():
          try:

              if(file_serializer.validated_data["modelCategory"]=="S2"):
                  print("S1_S2")
                  res = predictS2(file_serializer.validated_data["file"])
              elif(file_serializer.validated_data["modelCategory"]=="S1"):
                  print("S1_S2")
                  res = predictS2(file_serializer.validated_data["file"])
              else:
                  res = predictS1_S2(file_serializer.validated_data["file"])
                  print("S1_S2")
              d = {'result': res}
              print("Everything Executed Successfully")
              return Response(d, status=status.HTTP_201_CREATED)
          except Exception as e:
              predictions = {"error": "2", "message": str(e)}
              return Response(predictions, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
