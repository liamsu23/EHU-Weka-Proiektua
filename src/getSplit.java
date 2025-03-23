import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import java.io.File;

public class getSplit {
    public static void main(String[] args) throws Exception {
        // Archivo de entrada y salida
        String inTrainPath = args[0];  // Dataset original
        String outTrainSplitPath = args[1]; // Archivo de entrenamiento
        String outDevSplitPath = args[2]; // Archivo de validación

        // Cargar dataset
        DataSource source = new DataSource(inTrainPath);
        Instances data = source.getDataSet();

        // Establecer el índice de la clase (último atributo)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Crear filtro para dividir de forma estratificada (80% train - 20% dev)
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
        filter.setNumFolds(5); // 5-fold, 1 fold será dev (20%)
        filter.setFold(1); // Tomamos el primer fold como dev
        filter.setInvertSelection(false); // Filtramos para obtener solo dev
        filter.setInputFormat(data);
        Instances devData = weka.filters.Filter.useFilter(data, filter);

        filter.setInvertSelection(true); // Ahora seleccionamos el 80% restante como train
        Instances trainData = weka.filters.Filter.useFilter(data, filter);


        // Guardar train_split.arff
        ArffSaver trainSaver = new ArffSaver();
        trainSaver.setInstances(trainData);
        trainSaver.setFile(new File(outTrainSplitPath));
        trainSaver.writeBatch();

        // Guardar dev_split.arff
        ArffSaver devSaver = new ArffSaver();
        devSaver.setInstances(devData);
        devSaver.setFile(new File(outDevSplitPath));
        devSaver.writeBatch();

        System.out.println("División completada: Train → " + trainData.numInstances() + " instancias, Dev → " + devData.numInstances());
    }
}
