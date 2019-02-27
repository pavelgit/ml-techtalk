db.getCollection("set_1151").drop();

db.getCollection("mm_tariffqueryresults").find({}).forEach(function(doc){
    var tariffData = doc.tariffs.find(t => t.id === 1151);
    var obj = {
        _id: doc._id,
        created: doc.created
    };
    if (tariffData) {
        obj.output = {
            exists: true,
            net: tariffData.netPrice,
            gross: tariffData.grossPrice
        };
    } else {
        obj.output = { 
            exists: false
        };
    }

    obj.input = {
        occupation: doc.calculationInput.occupation,
        familyStatus: doc.calculationInput.familyStatus,
        birthday: ISODate(doc.calculationInput.birthday),
        educationType: doc.calculationInput.educationType,
        jobSituation: doc.calculationInput.jobSituation,
        benefitAgeLimit: Number(doc.calculationInput.benefitAgeLimit),
        benefitAmount: Number(doc.calculationInput.benefitAmount),
        fractionOfficeWork: Number(doc.calculationInput.fractionOfficeWork),
        industry: doc.calculationInput.industry,
        staffResponsibility: Number(doc.calculationInput.staffResponsibility),
        smoker: doc.calculationInput.smoker,
        insuranceStart: ISODate(doc.calculationInput.insuranceStart),
    };

    obj.inputHash = hex_md5(JSON.stringify(obj.input));
    obj.outputHash = hex_md5(JSON.stringify(obj.output));

    db.getCollection("set_1151").insert(obj);
});