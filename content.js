function removeChatElement() {
    let xpathChat = '/html/body/div/div/div/div/div[2]/aside';
    let iterator = document.evaluate(xpathChat, document, null, XPathResult.UNORDERED_NODE_ITERATOR_TYPE, null);
    let elementChat = iterator.iterateNext();

    if (elementChat) {
        elementChat.style.display = 'none';
    }
}

function changeLiveViewAttr() {
    let xpathLiveView = '/html/body/div/div/div/div/main/div';
    let iteratorLiveView = document.evaluate(xpathLiveView, document, null, XPathResult.UNORDERED_NODE_ITERATOR_TYPE, null);
    let elementLiveView = iteratorLiveView.iterateNext();

    if (elementLiveView) {
        elementLiveView.style.paddingRight = '0px';
    }
}

removeChatElement()
changeLiveViewAttr()