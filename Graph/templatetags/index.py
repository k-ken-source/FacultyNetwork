from django import template

register = template.Library()

def Find(indexable, i):
    return indexable[i]

register.filter('Find',Find)

