import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;


public class Baseline {

    public static void main(String[] args) throws Exception {

        if (args.length < 2) {
            System.out.println("Uso: java BaselineRegression <train.arff> <dev.arff>");
            return;
        }

        try {
            String trainPath = args[0];
            String devPath = args[1];

            // Cargar datasets
            ConverterUtils.DataSource sourceTrain = new ConverterUtils.DataSource(trainPath);
            Instances dataTrain = sourceTrain.getDataSet();

            ConverterUtils.DataSource sourceDev = new ConverterUtils.DataSource(devPath);
            Instances dataDev = sourceDev.getDataSet();

            // Establecer la variable objetivo (última columna)
            if (dataTrain.classIndex() == -1) {
                dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
            }
            if (dataDev.classIndex() == -1) {
                dataDev.setClassIndex(dataDev.numAttributes() - 1);
            }

            // 3. Crear el modelo de Regresión Logística
            Classifier model = new SimpleLogistic();
            //Classifier model = new Logistic();
            System.out.println("Entrenando modelo...");
            model.buildClassifier(dataTrain);

            // 4. Evaluar el modelo con el conjunto de desarrollo (dev)
            Evaluation eval = new Evaluation(dataTrain);
            System.out.println("Evaluando modelo...");
            eval.evaluateModel(model, dataDev);


            // 5. Imprimir resultados de la evaluación
            System.out.println("=== Evaluación del Modelo con Dev ===");
            System.out.println(eval.toSummaryString());
            System.out.println("=== Matriz de Confusión ===");
            System.out.println(eval.toMatrixString());
            System.out.println("=== Medidas de Precisión ===");
            System.out.println(eval.toClassDetailsString());

            // Imprimir métricas de regresión
            System.out.println("Error absoluto medio (MAE): " + eval.meanAbsoluteError());
            System.out.println("Raíz del error cuadrático medio (RMSE): " + eval.rootMeanSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }
}
