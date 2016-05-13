package neuralnetwork;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Artem on 11.05.2016.
 */
public class NeuralNetwork {
    private List<Layer> layers;
    private List<Double> output;

    private final static int NUMBER_OF_EPOCHS = 500000;
    private final static double ERROR_TOLERANCE = 0.01;

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

    public void setBiases(List<Double> biases) {
        if (biases == null)
            throw new NullPointerException("NeuralNetwork: biases can't be null");

        if (biases.size() != layers.size())
            throw new IllegalArgumentException("NeuralNetwork: biases size should be equal number of layers");

        for (int i = 0; i < layers.size(); ++i)
            layers.get(i).setBias(biases.get(i));
    }

    public void setBias(double bias, int layer) {
        layers.get(layer).setBias(bias);
    }

    public void loadNetwork(String fileAddress) {
        throw new NotImplementedException();
    }

    public void saveNetwork(String fileAddress) {
        throw new NotImplementedException();
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
