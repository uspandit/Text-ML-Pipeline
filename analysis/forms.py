from django import forms

class UploadFileForm(forms.Form):
    csv_file = forms.FileField()
    ngram_range = forms.ChoiceField(choices=[(1, 'Unigram'), (2, 'Bigram'), (3, 'Trigram')])
    analyzer = forms.ChoiceField(choices=[('word', 'Word'), ('char', 'Character')])
