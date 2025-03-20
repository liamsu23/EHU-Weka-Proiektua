import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.File;

public class arff2bow {
    public static void main(String[] args) {
        // Rutas de archivos
        String inTrain = "data/train_Zikina.arff";
        String outTrainBOW = "data/trainBOW.arff";
        String outHiztegia = "data/hiztegia.arff";

        try {
            // Cargar el archivo ARFF de entrada
            DataSource source = new DataSource(inTrain);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            // Usar el filtro StringToWordVector (Bag of Words)
            StringToWordVector bowFilter = new StringToWordVector();
            bowFilter.setInputFormat(data); // Especificar el formato de entrada
            bowFilter.setLowerCaseTokens(true); // Convertir a minúsculas
            bowFilter.setOutputWordCounts(true); // Usar frecuencias de palabras (no solo presencia/ausencia)
            bowFilter.setWordsToKeep(2000); // Limitar el número de palabras en el vocabulario

            // Aplicar el filtro para convertir el texto en BoW
            Instances bowData = Filter.useFilter(data, bowFilter);

            // Convertir de Sparse a Non-Sparse
            SparseToNonSparse sparseFilter = new SparseToNonSparse();
            sparseFilter.setInputFormat(bowData);
            Instances bowNonSparseData = Filter.useFilter(bowData, sparseFilter);

            // Pasarle al cliente el diccionario vacio
            Instances train_hutsik = new Instances(bowNonSparseData);
            train_hutsik.delete();

            // Hiztegia gorde
            ArffSaver arffSaver1 = new ArffSaver();
            arffSaver1.setFile(new File(outHiztegia));
            arffSaver1.setInstances(train_hutsik);
            arffSaver1.writeBatch();

            // Guardar el resultado en un nuevo archivo ARFF
            ArffSaver arffSaver2 = new ArffSaver();
            arffSaver2.setFile(new File(outTrainBOW));
            arffSaver2.setInstances(train_hutsik);
            arffSaver2.writeBatch();

            System.out.println("Archivo BOW generado exitosamente en: " + outTrainBOW);
            System.out.println("Archivo diccionario generado exitosamente en: " + outHiztegia);


        } catch (Exception e) {
            e.printStackTrace(); // Manejo de errores
        }
    }
}
