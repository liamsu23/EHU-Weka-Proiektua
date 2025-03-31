import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.PrintWriter;

public class getBaselineLinearRegression {
    public static void main(String[] args) {
        try {
            // Argumentuak kargatu
            String inTrain = args[0];
            String inDev = args[1];
            String inTest = args[2];

            String emaitzakPath = args[3];
            String outModel = args[4];
            String outIragarpenak = args[5];


            // Datuak kargatu
            ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(inTrain);
            Instances train = source1.getDataSet();
            train.setClassIndex(0);

            ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(inDev);
            Instances dev = source2.getDataSet();
            dev.setClassIndex(0);

            ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(inTest);
            Instances test = source3.getDataSet();
            test.setClassIndex(0);


            // Modeloa entrenatu
            LinearRegression model = new LinearRegression();
            model.buildClassifier(train);

            // Modeloa ebaluatu
            Evaluation evalDev = new Evaluation(train);
            evalDev.evaluateModel(model, dev);

            Evaluation evalTest = new Evaluation(train);
            evalTest.evaluateModel(model, test);

            // Emaitzak gorde
            saveResults(emaitzakPath, evalDev, evalTest);

            // Modeloa gorde
            SerializationHelper.write(outModel, model);


            savePredictions(outIragarpenak, model, test);


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void saveResults(String filename, Evaluation evalDev, Evaluation evalTest) throws Exception {
        PrintWriter writer = new PrintWriter(new FileWriter(filename));
        writer.println("Resultados en Dev:");
        writer.println("MAE: " + evalDev.meanAbsoluteError());
        writer.println("RMSE: " + evalDev.rootMeanSquaredError());
        writer.println("R^2: " + evalDev.correlationCoefficient());

        writer.println("\nResultados en Test:");
        writer.println("MAE: " + evalTest.meanAbsoluteError());
        writer.println("RMSE: " + evalTest.rootMeanSquaredError());
        writer.println("R^2: " + evalTest.correlationCoefficient());

        writer.close();
        System.out.println("Resultados guardados en '" + filename + "'");
    }

    private static void savePredictions(String filename, LinearRegression model, Instances testData) throws Exception {
        PrintWriter writer = new PrintWriter(new FileWriter(filename));
        writer.println("Índice\tReal\tPredicción");

        for (int i = 0; i < testData.numInstances(); i++) {
            double realValue = testData.instance(i).classValue();  // Valor real
            double predictedValue = model.classifyInstance(testData.instance(i));  // Predicción

            writer.println(i + "\t" + realValue + "\t" + predictedValue);
        }
        writer.close();
    }
}