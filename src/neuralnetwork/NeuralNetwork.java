package neuralnetwork;

import java.io.*;
import java.util.*;

/**
 * Created by Artem on 11.05.2016.
 */
public class NeuralNetwork {
    private List<Layer> layers;
    private List<Double> output;

    private final static int NUMBER_OF_EPOCHS = 10000;
    private final static double ERROR_TOLERANCE = 0.01;

    public NeuralNetwork() {
        layers = new ArrayList<>();
        output = new ArrayList<>();
    }

    public NeuralNetwork(List<Integer> neuronsInLayers) {
        if (neuronsInLayers.size() < 2) {
            throw new IllegalArgumentException("NeuralNetwork: network can't have less than two layers");
        }

        layers = new ArrayList<>();
        output = new ArrayList<>();

        for (int i = 1; i < neuronsInLayers.size(); ++i) {
            layers.add(new Layer(neuronsInLayers.get(i), neuronsInLayers.get(i - 1)));
        }

        layers.get(layers.size() - 1).setAsOutputLayer();
    }

    public int classify(List<Double> inputs) {
        if (inputs == null) {
            throw new NullPointerException("NeuralNetwork: inputs can't be null");
        }

        for (Layer layer : layers) {
            inputs = layer.feedForward(inputs);
        }

        output = new ArrayList<>(inputs);
        return getClassID();
    }

    public void trainNetwork(List<List<Double>> inputs, List<List<Double>> correctAnswers, double alpha) {
        if (inputs == null) {
            throw new NullPointerException("NeuralNetwork: inputs can't be null");
        }

        if (correctAnswers == null) {
            throw new NullPointerException("NeuralNetwork: correctAnswers can't be null");
        }

        if (inputs.size() != correctAnswers.size()) {
            throw new IllegalArgumentException("NeuralNetwork: inputs and correctAnswers should be of the same size");
        }

        Random random = new Random();

        for (int i = 0; i < NUMBER_OF_EPOCHS; ++i) {
            int inputIndex = random.nextInt(inputs.size());

            classify(inputs.get(inputIndex));

            List<Double> networkOutput = getOutputs();
            List<Double> correctAnswer = correctAnswers.get(inputIndex);

            if (i % 1000 == 0) {
                System.out.println("\nEpoch " + i + ":\n" + toString());
                System.out.println(inputs.get(inputIndex) + " => " + networkOutput + ". Correct answer: " + correctAnswer);
            }

            double error = getError(networkOutput, correctAnswer);

            /*if (error < ERROR_TOLERANCE)
                break;*/

            backPropagate(networkOutput, correctAnswer, alpha);
        }
    }

    public void testNetwork(List<List<Double>> inputs, List<List<Double>> correctAnswers) {
        if (inputs == null) {
            throw new NullPointerException("NeuralNetwork: inputs can't be null");
        }

        if (correctAnswers == null) {
            throw new NullPointerException("NeuralNetwork: correctAnswers can't be null");
        }

        if (inputs.size() != correctAnswers.size()) {
            throw new IllegalArgumentException("NeuralNetwork: inputs and correctAnswers should be of the same size");
        }

        System.out.println("\n --------- TEST ---------\n");

        int totalInputsCounter = 0;
        int correctAnswersCounter = 0;

        for (int i = 0; i < inputs.size(); ++i) {
            int classID = classify(inputs.get(i));
            int correctClass = 0;

            for (int j = 0; j < correctAnswers.get(i).size(); ++j) {
                if (correctAnswers.get(i).get(j) == 1.0)
                    correctClass = j;
            }

            System.out.println(inputs.get(i) + " => " + classID + ". Correct answer: " + correctClass);

            if (classID == correctClass)
                ++correctAnswersCounter;

            ++totalInputsCounter;
        }

        double accuracy =  1.0 * correctAnswersCounter / totalInputsCounter;

        System.out.println("\nAccuracy: " + accuracy);
    }

    public int getClassID() {
        double maxOutput = Double.MIN_VALUE;
        int classID = -1;

        for (int i = 0; i < output.size(); ++i) {
            if (output.get(i) > maxOutput) {
                maxOutput = output.get(i);
                classID = i;
            }
        }

        return classID;
    }

    List<Double> getOutputs() {
        return this.output;
    }

    public List<List<Double>> getLayerWeights(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.size()) {
            throw new IllegalArgumentException("NeuralNetwork: layerIndex is out of range");
        }

        return layers.get(layerIndex).getWeights();
    }

    public void setLayerWeights(int layer, List<List<Double>> weights) {
        if (weights == null)
            throw new NullPointerException("NeuralNetwork: weights can't be null");

        if (layer < 0 || layer >= layers.size())
            throw new IllegalArgumentException("NeuralNetwork: index of layer is out of range");

        layers.get(layer).setWeights(weights);
    }

    public void setBiases(List<List<Double>> biases) {
        if (biases == null)
            throw new NullPointerException("NeuralNetwork: biases can't be null");

        if (biases.size() != layers.size())
            throw new IllegalArgumentException("NeuralNetwork: biases size should be equal number of layers");

        for (int i = 0; i < layers.size(); ++i)
            layers.get(i).setBiases(biases.get(i));
    }

    public void addLayer(Layer layer) {
        if (layer == null) {
            throw new NullPointerException("NeuralNetwork: layer can't be null");
        }

        if (layers.size() > 1) {
            Layer previousLayer = layers.get(layers.size() - 1);
            previousLayer.setAsHiddenLayer();
        }

        layer.setAsOutputLayer();

        layers.add(layer);
    }

    public void addLayer(int numberOfNeurons) {
        Layer previousLayer = layers.get(layers.size() - 1);
        previousLayer.setAsHiddenLayer();

        Layer newLayer = new Layer(numberOfNeurons, previousLayer.size());
        newLayer.setAsOutputLayer();

        layers.add(newLayer);
    }

    public void setBiases(int layerIndex, List<Double> biases) {
        if (layerIndex < 0 || layerIndex >= layers.size())
            throw new IllegalArgumentException("NeuralNetwork: index of layer is out of range");

        layers.get(layerIndex).setBiases(biases);
    }

    public List<Double> getBiases(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.size())
            throw new IllegalArgumentException("NeuralNetwork: index of layer is out of range");

        return layers.get(layerIndex).getBiases();
    }

    public double getBias(int layerIndex, int neuronIndex) {
        if (layerIndex < 0 || layerIndex >= layers.size())
            throw new IllegalArgumentException("NeuralNetwork: index of layer is out of range");

        return layers.get(layerIndex).getBias(neuronIndex);
    }

    public int size() {
        return this.layers.size();
    }

    public static NeuralNetwork loadNetwork(String fileAddress) throws IOException {
        File file = new File(fileAddress);
        InputStream inp = new FileInputStream(file);
        Scanner scanner = new Scanner(inp);

        NeuralNetwork neuralNetwork = new NeuralNetwork();

        int numberOfLayers;
        List<Integer> neuronsInLayers = new ArrayList<>();

        if (!scanner.hasNextLine()) {
            throw new IOException("NeuralNetwork: file is empty");
        } else {
            String line = scanner.nextLine();
            numberOfLayers = Integer.parseInt(line.trim());
        }

        if (!scanner.hasNextLine()) {
            throw new IOException("NeuralNetwork: data is missed. Should contain numbers of neurons in each layer");
        } else {
            String[] neuronsNumbers = scanner.nextLine().split(" ");

            for (String numberOfNeurons : neuronsNumbers) {
                neuronsInLayers.add(Integer.parseInt(numberOfNeurons.trim()));
            }
        }

        int layerIndex = 1;
        StringBuilder sLayer = new StringBuilder();

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();

            if (line.equals("#")) {

                if (sLayer.length() != 0) {
                    Layer newLayer = Layer.parseLayer(sLayer.toString());

                    if (newLayer.size() != neuronsInLayers.get(layerIndex)) {
                        throw new IOException("NeuralNetwork: inconveniences in file data.");
                    }

                    neuralNetwork.addLayer(newLayer);
                    ++layerIndex;
                }

                sLayer = new StringBuilder();
            } else {
                sLayer.append(line + "\n");
            }
        }

        if (sLayer.length() > 0) {
            Layer newLayer = Layer.parseLayer(sLayer.toString());

            if (newLayer.size() != neuronsInLayers.get(layerIndex)) {
                throw new IOException("NeuralNetwork: inconveniences in file data.");
            }

            neuralNetwork.addLayer(newLayer);
        }

        if (neuralNetwork.size() + 1 != numberOfLayers) {
            throw new IOException("NeuralNetwork: inconveniences in file data.");
        }

        return neuralNetwork;
    }

    public void saveNetwork(String fileAddress) throws IOException {
        Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fileAddress), "utf-8"));

        writer.write((layers.size() + 1) + "\n");

        Integer inputLayerSize = layers.get(0).inputSize();
        writer.write(inputLayerSize.toString());

        for (Layer layer : layers) {
            writer.write(" " + layer.size());
        }

        for (Layer layer : layers) {
            writer.write("\n#\n");
            writer.write(layer.toString());
        }

        writer.close();
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder();

        text.append("Network:\n");

        for (Layer layer : layers) {
            text.append(layer);
        }

        return text.toString();
    }

    private void backPropagate(List<Double> networkAnswer, List<Double> correctAnswer, double alpha) {
        layers.get(layers.size() - 1).backPropagateOutputLayer(networkAnswer, correctAnswer, alpha);

        for (int i = layers.size() - 2; i >= 0; --i) {
            List<List<Double>> nextLayerWeights = layers.get(i + 1).getWeights();
            List<Double> nextLayerSigmas = layers.get(i + 1).getSigmas();

            layers.get(i).backPropagate(nextLayerWeights, nextLayerSigmas, alpha);
        }

        for (Layer layer : layers) {
            layer.updateWeights();
        }
    }

    private double getError(List<Double> networkAnswer, List<Double> correctAnswer) {
        if (correctAnswer.size() != networkAnswer.size())
            throw new IllegalArgumentException("NeuralNetwork: networkAnswer and correctAnswer should be of the same size");

        double error = 0;

        for (int i = 0; i < networkAnswer.size(); ++i)
            error += (correctAnswer.get(i) - networkAnswer.get(i)) * (correctAnswer.get(i) - networkAnswer.get(i));

        return Math.sqrt(error);
    }
}
