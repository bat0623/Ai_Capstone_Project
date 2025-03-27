import os, json, re, sys
import parsers
from copy import deepcopy


def writeData(out,folder):
	json_data = json.dumps({"text":out}, indent="\t",ensure_ascii=False)
	# make a bit more compact
	json_data = re.sub('{\n\t+','{',json_data)
	json_data = re.sub('\n\t+}','}',json_data)
#		json_data = re.sub('\[\n\t+','[',json_data)
	json_data = re.sub('\n\t+]',']',json_data)
	# Non-dialogue info should not have its own line
	json_data = re.sub('",\n\t+"_','", "_',json_data)
	o = open(folder+"data.json",'w')
	o.write(json_data)
	o.close()
	
def changeAliasOneLevel(dic,ali):
	if isinstance(dic,dict):
		for old_key in list(dic):
			if(old_key in ali.keys()):
				replaceName = ali[old_key]
				if isinstance(replaceName,str):
					# (alias can be list of names to split into)
					# We're creating a new rep, so we can preserve the order
					#  (yes, I know dictionaries are not ordered, but this does 
					#   make a difference to the output)
					rep = {replaceName:dic[old_key]}#
					for otherKey in [key for key in dic.keys() if key.startswith("_")]:# 
						rep[otherKey] = dic[otherKey]#
					dic = rep
					#dic[replaceName] = dic.pop(old_key)
				elif isinstance(replaceName,dict):
					# A dictionary that matches by line
					dialogue = dic[old_key]
					candidates = [kx for kx,vx in replaceName.items() if any([dialogue.startswith(y) for y in vx])]
					if len(candidates)>0:
						rep = {candidates[0]:dic[old_key]}#
						for otherKey in [key for key in dic.keys() if key.startswith("_")]:# 
							rep[otherKey] = dic[otherKey]#
						dic = rep
						#dic[candidates[0]] = dic.pop(old_key)
	return(dic)

	
def changeAliasMulitlevel_applyMetaFileOrder(lines,aliases):
	# For one line, replace a single string alias
	def replaceStringAlias(dic,target,replacement):
		# Create new dictionary, then put other keys after,
		#  to preserve order
		rep = {replacement:dic[target]}
		for otherKey in [key for key in dic.keys() if key.startswith("_")]:
			rep[otherKey] = dic[otherKey]
		return(rep)
	
	# For one line, replace a single alias (string, list or dictionary)
	# Returns list of lines (in case of splitting)
	def replaceOneAlias(dic,target,replacement):
		if isinstance(replacement, str):
			return([replaceStringAlias(dic,target,replacement)])
		elif isinstance(replacement, list):
			out = []
			for rep in replacement:
				out.append(replaceStringAlias(dic,target,rep))
			return(out)
		elif isinstance(replacement, dict):
			dlg = dic[target]
			for replName in replacement:
				if any([dlg.startswith(x) for x in replacement[replName]]):
					return([replaceStringAlias(dic,target,replName)])
			# No changes
			return([dic])
	
	# For all lines, apply one alias transformation
	def applyOneAlias(lines,target,replacement):
		out = []
		for line in lines:
			if "CHOICE" in line:
				line["CHOICE"] = [applyOneAlias(x,target,replacement) for x in line["CHOICE"]]
				out.append(line)
			else:
				if target in line:
					out += replaceOneAlias(line,target,replacement)
				else:
					out.append(line)
		return(out)
	# Main loop for changeAliasMulitlevel_applyMetaFileOrder
	if len(aliases)==0:
		return(lines)
	for target in aliases:
		replacement = aliases[target]
		lines = applyOneAlias(lines,target,replacement)
	return(lines)
	
folders = [root+os.sep for root,dirs,files in os.walk("../data/") if "meta.json" in files]

# Allow parsing of just one game
if len(sys.argv)>1:
	fx = sys.argv[-1]
	if not fx.endswith(os.sep):
		fx += os.sep
	folders = [fx]

for folder in folders:
	with open(folder+"meta.json") as json_file:
		meta = json.load(json_file)
	pp = meta["parserParameters"]
	print("PARSING "+meta["game"])

	parseMethod = getattr(getattr(parsers, pp["parser"]),"parseFile")
	fileType = "html"
	if "fileType" in pp:
		fileType = pp["fileType"]
	rawFiles = [x for x in os.listdir(folder+"raw/") if x.endswith(fileType)]
	rawFiles.sort()
	out = []
	for rawFile in rawFiles:
		out += parseMethod(folder+"raw/"+rawFile,pp)
	
	if hasattr(getattr(parsers, pp["parser"]), 'postProcessing'):
		postProcessingMethod = getattr(getattr(parsers, pp["parser"]),"postProcessing")
		out = postProcessingMethod(out)
	
	#altAliasOut = deepcopy(out)
	if "aliases" in meta.keys():
		# At the beginning of the project, the alias algorithm iterated over lines, applying all rules to each line.
		#  However, coders were creating rules as if they iterated over rules, applying a rule to all lines.
		#  So another algorithm was implemented
		#out = changeAliasMulitlevel(out,meta["aliases"])
		out = changeAliasMulitlevel_applyMetaFileOrder(out,meta["aliases"])
	
	writeData(out,folder)

