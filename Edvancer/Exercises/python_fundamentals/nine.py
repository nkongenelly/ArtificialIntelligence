def fun_nine(sentence,stop_words):
    new_sentence = [i for i in sentence.split(' ') if i not in stop_words]
    # new_sentence = ' '.join(new_sentence)
    output = {}
    for word in new_sentence:
        output.update({word: new_sentence.count(word)})

    return output

if __name__ == '__main__':
    sentence = 'we must indeed all hang together or most assuredly we shall all hang separately'
    stop_words =  {'we','or','all','shall'}
    res = fun_nine(sentence,stop_words)
    print(res)