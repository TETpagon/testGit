from developTools import toolsFile
from Class.RepositorySignals import RepositorySignals

if __name__ == "__main__":
    # dataFromXML = researchFilesSignals.getDataFromFilesSignals()
    # researchFilesSignals.writeToPickle(config.pathToDataFromXML, dataFromXML)
    #
    # # dataFromXML = researchFilesSignals.readeFromPickle(config.pathToDataFromXML)
    #
    # valuesSignals = researchFilesSignals.getValuesSignal(dataFromXML)
    # del(dataFromXML)
    # researchFilesSignals.writeToPickle(config.pathToValuesSignals, valuesSignals)
    #
    # amountValuesInSignal = researchFilesSignals.getAmountValuesInSignal(valuesSignals)
    # del(valuesSignals)
    # researchFilesSignals.writeToPickle(config.pathToAmountValuesInSignal, amountValuesInSignal)
    #
    # minMaxAmountValueInSignal = researchFilesSignals.getMinMaxAmountValueInSignal(amountValuesInSignal)
    # del(amountValuesInSignal)
    # researchFilesSignals.writeToPickle(config.pathToMinMaxAmountValueInSignal, minMaxAmountValueInSignal)

    # toolsFile.saveXmlFilesToPickle()

    repo = RepositorySignals()
    values = {}
    for index in range(repo.getAmountSignal()):
        signal = repo.getNextSignal()
        value = repo.getValuesSignal(signal)
        if (value['fTickDuration'] or value['fTickDuration-Data']) and value['fAcceleration'] and value['fForce'] and \
                value['fPosition']:
            values['values-' + signal['filename']] = value

        # if value['fAcceleration'] is not None and value['fForce'] is not None:
        #     values['values-' + signal['filename']] = value
    toolsFile.saveValuesSignalToPickle(values)
