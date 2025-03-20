import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

public class arff2bow {
    public static void main(String[] args) throws Exception {
        // Verificar que se han pasado los argumentos correctamente
        if (args.length < 3) {
            System.out.println("Erabilera: java arff2bow <train_raw.arff> <train_BoW.arff> <hiztegia.txt>");
            return;
        }

        try {
            // Asignar los parámetros de entrada
            String trainRawPath = args[0];  // Ruta al archivo .arff original
            String trainBoWPath = args[1];  // Ruta para guardar el archivo BoW procesado
            String dictionaryPath = args[2];  // Ruta para guardar el diccionario de palabras

            // Cargar el archivo de datos crudos
            DataSource source = new DataSource(trainRawPath);
            Instances trainData = source.getDataSet();

            // Establecer el índice de la clase (si no está definido)
            if (trainData.classIndex() == -1) {
                trainData.setClassIndex(trainData.numAttributes()-1);  // Se asume que la primera columna es la clase
            }

            // Configurar el filtro StringToWordVector para convertir los textos en Bag of Words (BoW)
            StringToWordVector filter = new StringToWordVector();
            // Ez da hiztegia murriztu behar, beraz hurrengo lerroa ez da beharrezkoa
            //filter.setWordsToKeep(1000);  // Limitar a las 1000 palabras más frecuentes
            filter.setOutputWordCounts(false);  // No contar la frecuencia de palabras
            filter.setLowerCaseTokens(true);  // Convertir toodo a minúsculas
            filter.setIDFTransform(false);  // No usar IDF (Inverse Document Frequency)
            filter.setTFTransform(false);  // No usar TF (Term Frequency)
            filter.setDictionaryFileToSaveTo(new File(dictionaryPath));  // Guardar el diccionario de palabras
            filter.setInputFormat(trainData);  // Establecer el formato de entrada del filtro

            // Aplicar el filtro para obtener el conjunto de datos en formato BoW
            Instances trainBoW = Filter.useFilter(trainData, filter);

            // Guardar el conjunto de datos BoW usando ArffSaver
            ArffSaver saver = new ArffSaver();
            saver.setInstances(trainBoW);  // Asignar el dataset procesado
            saver.setFile(new File(trainBoWPath));  // Especificar el archivo de salida
            saver.writeBatch();  // Guardar el archivo en formato ARFF

            System.out.println("BOW artxiboa gordeta: " + trainBoWPath);
            System.out.println("Hiztegia gordeta: " + dictionaryPath);

        } catch (Exception e) {
            // Capturar cualquier excepción y mostrar un mensaje de error
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }
}
