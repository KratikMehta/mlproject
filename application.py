import sys

from flask import Flask, render_template, request

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# Route for home page
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/predict-data", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender", "nan"),
            race_ethnicity=request.form.get("ethnicity", "nan"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education", "nan"
            ),
            lunch=request.form.get("lunch", "nan"),
            test_preparation_course=request.form.get("test_preparation_course", "nan"),
            reading_score=request.form.get("reading_score", 0, int),
            writing_score=request.form.get("writing_score", 0, int),
        )

        pred_df = data.get_data_as_dataframe()
        if isinstance(pred_df, Exception):
            raise CustomException(pred_df, sys.exc_info())

        pred_pipeline = PredictPipeline()
        results = pred_pipeline.predict(pred_df)
        if isinstance(results, Exception):
            raise CustomException(results, sys.exc_info())

        return render_template("home.html", results=round(float(results[0]), 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
